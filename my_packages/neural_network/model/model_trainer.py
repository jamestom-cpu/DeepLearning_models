from typing import Iterable
import os, sys, io
from contextlib import redirect_stdout



import torch
from torch import nn, optim
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

import mlflow
import mlflow.pytorch
import numpy as np
from my_packages.neural_network.model.early_stopping import EarlyStopping
from .aux_funcs_mlflow import clean_mlflow_metrics_efficient

class Trainer:
    def __init__(
            self, model, opt_func=torch.optim.SGD, 
            lr=0.01, patience=7, scheduler_kwargs={}, 
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
            _include_cleaning_of_mlflow_metrics=False,
            ):
        
        self.print_every_n_epochs = print_every_n_epochs
        self.model = model
        self.history = []
        self.lr = lr
        self.optimizer = opt_func(model.parameters(), lr)
        
        self.model_dir = model_dir

        if _include_cleaning_of_mlflow_metrics:
            self.clean_mlflow_metrics()

        self.early_stopping = EarlyStopping(patience=patience, path=os.path.join(self.model_dir, "checkpoint.pth"))
        self._init_optimizer_scheduler(**scheduler_kwargs)
        
        self.config = dict(
            lr=self.lr,
            opt_func=opt_func.__name__,
            patience=patience,
            damping_factor=scheduler_kwargs.get('factor', None),
        )

        self.config.update(parameters_of_interest)


        # setup mlflow logging
        self.experiment_name = experiment_name
        self.run_name = run_name
        # chose which to log
        self.log_mlflow = log_mlflow
        self.log_tensorboard = log_tensorboard
        # save each model to mlflow
        self.save_models_to_mlflow = save_models_to_mlflow


        #logging directories 
        self.mlflow_dir = os.path.join("/workspace", os.environ["MLFLOW_TRACKING_URI"])
        self.tensorboard_dir = os.environ["TENSORBOARD_LOG_DIR"]

        self.log_gradient = log_gradient
        self.log_weights = log_weights

        if self.log_mlflow:
            self._setup_mlflow_log()
        if self.log_tensorboard:
            self._setup_tensorboard_log()

    def clean_mlflow_metrics(self):
        clean_mlflow_metrics_efficient(self.mlflow_dir)
        

    def _setup_tensorboard_log(self):
        # tensorboard logging
        full_path_tensorboard_logdir = os.path.join(self.tensorboard_dir, self.experiment_name, self.run_name)
        self.writer = SummaryWriter(full_path_tensorboard_logdir)
    
    def _setup_mlflow_log(self):
        mlflow.set_experiment(self.experiment_name)
        self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
        mlflow.start_run(experiment_id=self.experiment_id, run_name=self.run_name)
        mlflow.log_params(self.config)
        # Log the summary string
        model_summary_str = self._summary_string()
        mlflow.log_text(model_summary_str, artifact_file="model_summary.txt")

    
    def _summary_string(self, device = "cpu"):
        f = io.StringIO()
        with redirect_stdout(f):
            summary(self.model, input_size=self.model.in_shape, device=device)
        return f.getvalue()


    def _init_optimizer_scheduler(self, **scheduler_kwargs):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, **scheduler_kwargs
            )
        return self
    
    def _train_on_batch(self, batch):
        loss = self.model.training_step(batch)
        loss.backward()
        self.optimizer.step()

        # log statistics
        # if self.log_mlflow:
        #     # mlflow
        #     self.log_gradient_statistics()
        #     self.log_weights_statistics()
        if self.log_tensorboard:
            #tensorboard
            self._log_gradient_histogram_tensorboard()
            self._log_weights_histogram_tensorboard()
        # clear gradients
        self.optimizer.zero_grad() 
        return loss
    

    def fit(self, epochs, train_loader, val_loader):
        for epoch in range(epochs):
            self.epoch = epoch
            self.model.train()
            train_losses = []
            for batch in train_loader:
                loss = self._train_on_batch(batch)
                train_losses.append(loss)
                

            # Validation phase
            self.model.eval() # set to evaluation mode
            result = self.model.evaluate(val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()

            # get quantities of interest
            train_loss = torch.stack(train_losses).mean().item()
            val_loss = result['val_loss']
            val_acc = result['val_acc']

            # dynamic learning rate adjustment
            self.scheduler.step(val_loss)
            
            # log history
            self._log_history(train_loss, val_loss, val_acc)
            #lof to mlflow
            if self.log_mlflow:
                self._log_metrics_mlflow(epoch, **self.history[-1])   
            # log to tensorboard
            if self.log_tensorboard:
                self._log_history_tensorboard(train_loss, val_loss, val_acc, epoch)

            if (epoch) % self.print_every_n_epochs == 0:
                self.model.epoch_end(epoch, result)

            # early stopping 
            self.early_stopping(val_loss, self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                # load the last checkpoint with the best model
                self.model.load_state_dict(torch.load('checkpoint.pth'))
                break
        
        # save model to mlflow and end the mlflow run
        if self.log_mlflow:
            if self.save_models_to_mlflow:
                mlflow.pytorch.log_model(self.model, "models")
            mlflow.end_run()
        
        # end tensorboard run
        if self.log_tensorboard:
            self.writer.close()
        return self.history

    def _log_history(self, train_loss, val_loss, val_acc):
        # get current learning rate
        lr = self.optimizer.param_groups[0]['lr']
        self.history.append({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': lr
        })

    def _log_history_tensorboard(self, train_loss, val_loss, val_acc, epoch):
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/validation', val_loss, epoch)
        self.writer.add_scalar('Accuracy', val_acc, epoch)


    def _log_param_statistics(self, name, param):
        param_data = param.detach().cpu().numpy()
        
        mean_val = np.mean(param_data)
        std_val = np.std(param_data)
        min_val = np.min(param_data)
        max_val = np.max(param_data)
        
        # Log parameter statistics
        mlflow.log_metric(f"{name}_mean", mean_val)
        mlflow.log_metric(f"{name}_std", std_val)
        mlflow.log_metric(f"{name}_min", min_val)
        mlflow.log_metric(f"{name}_max", max_val)


    def _log_param_histogram_tensorboard(self, name, param):
        self.writer.add_histogram(name, param, self.epoch)

    def _log_weights_statistics(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name.split('.')[0] in self.log_weights:
                self._log_param_statistics(name, param)

    def _log_gradient_statistics(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name.split('.')[0] in self.log_gradient:
                self._log_param_statistics(f"{name}_gradient", param.grad)

    def _log_gradient_histogram_tensorboard(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name.split('.')[0] in self.log_gradient:
                self._log_param_histogram_tensorboard(f"{name}_gradient", param.grad)
    
    def _log_weights_histogram_tensorboard(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name.split('.')[0] in self.log_weights:
                self._log_param_histogram_tensorboard(name, param)
        
    def _log_metrics_mlflow(self, epoch, train_loss, val_loss, val_acc, lr):
        if not all([isinstance(x, float) for x in [train_loss, val_loss, val_acc, lr]]):
            mlflow.end_run()
            raise ValueError("Metrics must be floats")
        mlflow.log_metrics({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': lr
        }, step=epoch)
