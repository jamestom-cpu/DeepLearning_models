from typing import Callable, Iterable, Tuple
import mlflow
import mlflow.pytorch
from tqdm import tqdm

import torch
import torch.nn as nn

from my_packages.neural_network.model.early_stopping import EarlyStopping
from my_packages.neural_network.model.model_trainers.trainer_base import Trainer_Base
from my_packages.neural_network.model.model_base import Model_Base




class Trainer(Trainer_Base):
    def __init__(
            self, 
            model: Model_Base, 
            opt_func=torch.optim.SGD, 
            loss_fn_binary=nn.BCEWithLogitsLoss(),
            loss_fn_magnitude=nn.MSELoss(),
            lr=0.01, patience=7, 
            scheduler_kwargs={}, 
            optimizer_kwargs={},
            model_dir="models", 
            log_mlflow=True,
            log_tensorboard=True,
            experiment_name="My Experiment",
            run_name=None,
            print_every_n_epochs=3,
            log_gradient: Iterable[str]=[], 
            log_weights: Iterable[str]=[],
            parameters_of_interest: dict={},
            save_models_to_mlflow=True,
            _include_cleaning_of_mlflow_metrics=False,):
        
        # initialize the loss functions
        self.loss_fn_binary = loss_fn_binary
        self.loss_fn_magnitude = loss_fn_magnitude

        # initialize the optimizer
        self.lr = lr
        self.model = model
        self.opt_func = opt_func
        self.scheduler_kwargs = scheduler_kwargs
        self.optimizer_kwargs = optimizer_kwargs

        self._init_optimizer(self.opt_func, lr, **self.optimizer_kwargs)
        self._init_optimizer_scheduler(**self.scheduler_kwargs)

        self.config = self._define_config_dict_of_interest(
            opt_func, patience, scheduler_kwargs, parameters_of_interest
            )
            
        super().__init__(model=model, patience=patience, model_dir=model_dir,
                         log_mlflow=log_mlflow, log_tensorboard=log_tensorboard,
                         experiment_name=experiment_name, run_name=run_name,
                         print_every_n_epochs=print_every_n_epochs,
                         log_gradient=log_gradient, log_weights=log_weights,
                         parameters_of_interest=parameters_of_interest,
                         save_models_to_mlflow=save_models_to_mlflow,
                         _include_cleaning_of_mlflow_metrics=_include_cleaning_of_mlflow_metrics
                         )
        
     

    def _define_config_dict_of_interest(
            self, opt_func, patience, scheduler_kwargs, 
            parameters_of_interest) -> dict:
        config = dict(
            lr=self.lr,
            opt_func=opt_func.__name__,
            patience=patience,
            damping_factor=scheduler_kwargs.get('factor', None),
        )
        config.update(parameters_of_interest)
        return config

    def _init_optimizer_scheduler(self, **scheduler_kwargs):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, **scheduler_kwargs
            )
        return self
    
    def _init_optimizer(self, opt_func: Callable, lr, **optimizer_kwargs):
        self.optimizer = opt_func(self.model.parameters(), lr, **optimizer_kwargs)

    def _train_on_batch_binary(self, batch):
        loss = self.model.training_step_dipole_position(batch, self.loss_fn_binary)
        loss.backward()
        self.optimizer.step()

        # log gradient statistics
        if self.log_tensorboard:
            #tensorboard
            self._log_gradient_histogram_tensorboard()
            self._log_weights_histogram_tensorboard()
        # clear gradients
        self.optimizer.zero_grad() 
        return loss.detach()
    
    def _train_on_batch_magnitude(self, batch):
        loss = self.model.training_step_dipole_magnitude(batch, self.loss_fn_magnitude)

        # clear gradients
        self.optimizer.zero_grad()

        # compute gradients
        loss.backward()
        self.optimizer.step()

        # log gradient statistics
        if self.log_tensorboard:
            #tensorboard
            self._log_gradient_histogram_tensorboard()
            self._log_weights_histogram_tensorboard()        
        return loss.detach()


    def fit_binary(self, epochs, train_loader, val_loader):
        self._prepare_for_training()
        
        for epoch in range(epochs):
            self.epoch = epoch
            self.model.train()
            train_losses = []
            for batch in tqdm(train_loader):
                loss = self._train_on_batch_binary(batch)
                train_losses.append(loss)
                
            result = self.model.evaluate_binary(val_loader, self.loss_fn_binary)
            result['train_loss'] = torch.stack(train_losses).mean().item()

            train_loss = torch.stack(train_losses).mean().item()
            val_loss = result['val_loss']
            val_acc = result['val_acc']

            self.scheduler.step(val_loss)
            
            self._log_history(train_loss, val_loss, val_acc)
            if self.log_mlflow:
                self._log_metrics_mlflow(epoch, **self.history[-1])   
            if self.log_tensorboard:
                self._log_history_tensorboard(train_loss, val_loss, val_acc, epoch)

            if (epoch) % self.print_every_n_epochs == 0:
                self.model.epoch_end(epoch, result)

            self.early_stopping(val_loss, self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                self.model.load_state_dict(torch.load(self.early_stopping_checkpoint_path))
                break
        
        if self.log_mlflow:
            print("Close mlflow session")
            if self.save_models_to_mlflow:
                mlflow.pytorch.log_model(self.model, "models")
            mlflow.end_run()
        
        if self.log_tensorboard:
            print("Close tensorboard session")
            self.writer.close()
        
        print("Training finished")
        return self.history
    
    def fit_magnitude(self, epochs, train_loader, val_loader):
        self._prepare_for_training()
        
        for epoch in range(epochs):
            self.epoch = epoch
            self.model.train()
            train_losses = []
            for batch in tqdm(train_loader):
                loss = self._train_on_batch_magnitude(batch)
                train_losses.append(loss)
                
            result = self.model.evaluate_magnitude(val_loader, self.loss_fn_magnitude)
            result['train_loss'] = torch.stack(train_losses).mean().item()

            train_loss = torch.stack(train_losses).mean().item()
            val_loss = result['val_loss']
            val_acc = result['val_acc']

            self.scheduler.step(val_loss)
            
            self._log_history(train_loss, val_loss, val_acc)
            if self.log_mlflow:
                self._log_metrics_mlflow(epoch, **self.history[-1])   
            if self.log_tensorboard:
                self._log_history_tensorboard(train_loss, val_loss, val_acc, epoch)

            if (epoch) % self.print_every_n_epochs == 0:
                self.model.epoch_end(epoch, result)

            self.early_stopping(val_loss, self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                self.model.load_state_dict(torch.load(self.early_stopping_checkpoint_path))
                break
        
        if self.log_mlflow:
            print("Close mlflow session")
            if self.save_models_to_mlflow:
                mlflow.pytorch.log_model(self.model, "models")
            mlflow.end_run()
        
        if self.log_tensorboard:
            print("Close tensorboard session")
            self.writer.close()
        
        print("Training finished")
        return self.history
    
    def reset_model_training(self):
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Reinitialize the optimizer
        self.optimizer = self.opt_func(self.model.parameters(), self.lr, **self.optimizer_kwargs)
        # Reinitialize the scheduler
        self._init_optimizer_scheduler(**self.scheduler_kwargs)
        # Reinitialize the early stopping
        self._init_early_stopping(patience=self.patience)
            
    
    def switch_to_magnitude(self):
        self._init_early_stopping(patience=self.patience)
        # Freeze the convolutional base
        for param in self.model.arch.conv_base.parameters():
            param.requires_grad = False
        # Unfreeze the magnitude head
        for param in self.model.arch.magntiude_head.parameters():
            param.requires_grad = True

        # Reinitialize the optimizer
        self.optimizer = self.opt_func(self.model.arch.magntiude_head.parameters(), self.lr, **self.optimizer_kwargs)
        # Reinitialize the scheduler
        self._init_optimizer_scheduler(**self.scheduler_kwargs)
    
    def fit(self, epochs, train_loader, val_loader):
        print("fitting func...")