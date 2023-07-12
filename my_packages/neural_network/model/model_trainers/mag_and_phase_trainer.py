from typing import Callable, Iterable, Tuple
# from numba import cuda
from tqdm import tqdm


import torch

import mlflow
import mlflow.pytorch


from .trainer_base import Trainer_Base
from my_packages.neural_network.model.model_base_mag_phase import Model_Base

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
        
        print("Trainer initialized")
        
        
        
        

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
        binary_loss, magnitude_loss, phase_loss = self.model.training_step(batch)
        # calculate the compounded loss
        total_loss = self.model.compounded_loss_fn(binary_loss, magnitude_loss, phase_loss)
        total_loss.backward()
        self.optimizer.step()

        # log gradient statistics
        if self.log_tensorboard:
            #tensorboard
            self._log_gradient_histogram_tensorboard()
            self._log_weights_histogram_tensorboard()
        # clear gradients
        self.optimizer.zero_grad() 
        return binary_loss.detach(), magnitude_loss.detach(), phase_loss.detach()
    
    
    def fit(self, epochs, train_loader, val_loader):
        self._prepare_for_training()
        
        for epoch in range(epochs):
            self.epoch = epoch
            # switch model to training mode
            self.model.train()
            # initialize the losses for the different components
            train_losses_binary = []
            train_losses_magnitude = []
            train_losses_phase = []


            for batch in tqdm(train_loader):
                binary_loss, magnitude_loss, phase_loss = self._train_on_batch(batch)
                train_losses_binary.append(binary_loss)
                train_losses_magnitude.append(magnitude_loss)
                train_losses_phase.append(phase_loss)
            

            result = self.model.evaluate(val_loader)
            result['train_loss_binary'] = torch.stack(train_losses_binary).mean().item()
            result['train_loss_magnitude'] = torch.stack(train_losses_magnitude).mean().item()
            result['train_loss_phase'] = torch.stack(train_losses_phase).mean().item()

            train_compounded_loss = self.model.compounded_loss_fn(
                result['train_loss_binary'], 
                result['train_loss_magnitude'], 
                result['train_loss_phase'])
            
            result['train_loss'] = train_compounded_loss.item()
            
            val_loss = result['val_loss']

            # take a step with the scheduler
            self.scheduler.step(val_loss)
            

            self._log_history(result)
            if self.log_mlflow:
                self._log_metrics_mlflow(epoch, **result)   
            if self.log_tensorboard:
                self._log_history_tensorboard(epoch, **result)

            if (epoch) % self.print_every_n_epochs == 0:
                self.model.epoch_end(epoch, result)

            self.early_stopping(val_loss, self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                self.model.load_state_dict(torch.load(self.early_stopping_checkpoint_path))
                break
    
    def _log_history(self, result):
        self.history.append(result)
    
    
    def _log_metrics_mlflow(self, epoch, **metrics):
        if not all([isinstance(x, float) for x in metrics.values()]):
            mlflow.end_run()
            raise ValueError("Metrics must be floats")
        metrics_with_step = {name: value for name, value in metrics.items()}
        mlflow.log_metrics(metrics_with_step, step=epoch)




    