from typing import Iterable, Callable
import os, sys, io
from contextlib import redirect_stdout
# from numba import cuda


import torch
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod

import mlflow
import mlflow.pytorch
import numpy as np
from my_packages.neural_network.model.early_stopping import EarlyStopping
from my_packages.neural_network.gpu_aux import to_device
from ..aux_funcs_mlflow import clean_mlflow_metrics_efficient

class Trainer_Base(ABC):
    def __init__(
            self, 
            model, 
            patience=3,
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
        
        
        self.model_dir = model_dir

        if _include_cleaning_of_mlflow_metrics:
            self.clean_mlflow_metrics()

        # initialize early stopping
        self._init_early_stopping(patience=patience)


        # initialize the config dictionary for hyperparameter tracking
        self.config = self._initialize_config_dict(parameters_of_interest=parameters_of_interest)


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

    @abstractmethod
    def _initialize_config_dict(self, **kwargs)->dict:
        """Initializes the config dictionary that is tracked by mlflow and tensorboard"""
        pass


    def clean_mlflow_metrics(self):
        """
        mlflow can through an error or not exhibit the visuals correctly if the metric
        files contains irregular entries. This can happen if the training is interrupted
        this function cleans the metrics files: 
        1. it walks through the mlflow directory until it finds metric files. If it does not find any it 
           returns

        2. it creates a temporary file

        3. it reads the metrics file line by line and writes the lines to the temporary file if the line
            contains exactly 3 values

        4. it replaces the original file with the temporary file

        5. it repeats the process for all metric files in the mlflow directory
        """
         
        clean_mlflow_metrics_efficient(self.mlflow_dir)

    def free_cuda_memory(self):
        # move model to cpu
        to_device(self.model, 'cpu')
        torch.cuda.empty_cache()
        
    # def _completely_free_cuda_memory_(self):
    #     """Can lead to segmentation fault"""
    #     to_device(self.model, 'cpu')
    #     torch.cuda.empty_cache()
    #     cuda.select_device(0)
    #     cuda.close()
        

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
            summary(self.model, input_size=self.model.input_shape, device=device)
        return f.getvalue()
    

    def _init_early_stopping(self, patience):
        """ 
        Default EarlyStopping Initialization: Simply overwrite this function in the child classes if needed
        
        """
        self.early_stopping_checkpoint_path = os.path.join(self.model_dir, "checkpoint.pth")
        self.early_stopping = EarlyStopping(patience=patience, path=self.early_stopping_checkpoint_path)
    
    def _init_optimizer(self, opt_func: Callable, **optimizer_kwargs):
        """ 
        Default Optimizer Initialization: Simply overwrite this function in the child classes if needed
        
        """
        self.optimizer = opt_func(self.model.parameters(), **optimizer_kwargs)


    def _init_optimizer_scheduler(self, **scheduler_kwargs):
        """ 
        Default Optimizer Scheduler Initialization: Simply overwrite this function in the child classes if needed
        
        """
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, **scheduler_kwargs
            )
        return self
    
    def _train_on_batch(self, batch):
        """ 
        Default Training Step: Simply overwrite this function in the child classes if needed
        
        """
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
        return loss
    
    def _prepare_for_training(self):
        # start by clearing the cuda memory
        torch.cuda.empty_cache()

        # check that the the model is on the GPU
        to_device(self.model, 'cuda')


    @abstractmethod
    def fit(self, *args, **kwargs):
        pass
    

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