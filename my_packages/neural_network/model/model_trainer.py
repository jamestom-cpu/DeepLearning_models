from typing import Iterable
import os, sys, io
from contextlib import redirect_stdout



import torch
from torch import nn, optim
import torch.nn.functional as F
from torchsummary import summary

import mlflow
import mlflow.pytorch
import numpy as np
from my_packages.neural_network.model.early_stopping import EarlyStopping

class Trainer:
    def __init__(
            self, model, opt_func=torch.optim.SGD, 
            lr=0.01, patience=7, scheduler_kwargs={}, 
            model_dir="models", 
            run_name=None,
            experiment_name="My Experiment",
            log_gradient: Iterable[str]=[], 
            log_weights: Iterable[str]=[],
            parameters_of_interest: dict={},
            ):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.model = model
        self.history = []
        self.lr = lr
        self.optimizer = opt_func(model.parameters(), lr)
        
        self.model_dir = model_dir

        self.early_stopping = EarlyStopping(patience=patience, path=os.path.join(self.model_dir, "checkpoint.pth"))
        self.init_optimizer_scheduler(**scheduler_kwargs)
        
        self.config = dict(
            lr=self.lr,
            opt_func=opt_func.__name__,
            patience=patience,
            damping_factor=scheduler_kwargs.get('factor', None),
        )

        self.config.update(parameters_of_interest)

        
        # setup mlflow logging
        # self.experiment_id = mlflow.create_experiment("My Experiment")
        mlflow.set_experiment(self.experiment_name)
        self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
        mlflow.start_run(experiment_id=self.experiment_id, run_name=self.run_name)
        mlflow.log_params(self.config)
        # Log the summary string
        model_summary_str = self.summary_string()
        mlflow.log_text(model_summary_str, artifact_file="model_summary.txt")
        
        self.log_gradient = log_gradient
        self.log_weights = log_weights

    
    def summary_string(self, device = "cpu"):
        f = io.StringIO()
        with redirect_stdout(f):
            summary(self.model, input_size=self.model.in_shape, device=device)
        return f.getvalue()


    def init_optimizer_scheduler(self, **scheduler_kwargs):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, **scheduler_kwargs
            )
        return self
    
    def _train_on_batch(self, batch):
        loss = self.model.training_step(batch)
        loss.backward()
        self.optimizer.step()
        self.log_gradient_statistics()
        self.log_weights_statistics()
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
            self.log_history(train_loss, val_loss, val_acc)
            self.log_metrics_mlflow(epoch, **self.history[-1])   

            if (epoch) % 5 == 0:
                self.model.epoch_end(epoch, result)

            # early stopping 
            self.early_stopping(val_loss, self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                # load the last checkpoint with the best model
                self.model.load_state_dict(torch.load('checkpoint.pth'))
                break
        # log model
        mlflow.pytorch.log_model(self.model, "models")
        # end mlflow run
        mlflow.end_run()
        return self.history

    def log_history(self, train_loss, val_loss, val_acc):
        # get current learning rate
        lr = self.optimizer.param_groups[0]['lr']
        self.history.append({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': lr
        })
    def log_param_statistics(self, name, param):
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


    def log_weights_statistics(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name.split('.')[0] in self.log_weights:
                self.log_param_statistics(name, param)

    def log_gradient_statistics(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name.split('.')[0] in self.log_gradient:
                self.log_param_statistics(f"{name}_gradient", param.grad)
        
    def log_metrics_mlflow(self, epoch, train_loss, val_loss, val_acc, lr):
        mlflow.log_metrics({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': lr
        }, step=epoch)
