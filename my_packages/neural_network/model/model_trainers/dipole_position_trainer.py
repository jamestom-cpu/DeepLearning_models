from typing import Callable, Iterable, Tuple
# from numba import cuda
from tqdm import tqdm


import torch

import mlflow
import mlflow.pytorch


from .trainer_base import Trainer_Base
from my_packages.neural_network.model.model_base import Model_Base

class Trainer(Trainer_Base):
    def __init__(
            self, 
            model: Model_Base, 
            opt_func=torch.optim.SGD, 
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
        
        self.lr = lr
        self.model = model
        self._init_optimizer(opt_func, lr, **optimizer_kwargs)
        self._init_optimizer_scheduler(**scheduler_kwargs)

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

    def _train_on_batch(self, batch):
        loss = self.model.training_step(batch)
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


    def fit(self, epochs, train_loader, val_loader):
        self._prepare_for_training()
        
        for epoch in range(epochs):
            self.epoch = epoch
            self.model.train()
            train_losses = []
            for batch in tqdm(train_loader):
                loss = self._train_on_batch(batch)
                train_losses.append(loss)
                
            result = self.model.evaluate(val_loader)
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