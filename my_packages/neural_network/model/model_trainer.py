import torch
from torch import nn, optim
import torch.nn.functional as F
import os
import mlflow

from my_packages.neural_network.model.early_stopping import EarlyStopping

class Trainer:
    def __init__(
            self, model, opt_func=torch.optim.SGD, lr=0.01, 
            patience=7, lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,  
            scheduler_kwargs={}, log_to_wandb=True, model_dir="models", print_interval=3,
            wandb_name ="CNN"
            ):
        self.model = model
        self.history = []
        self.lr = lr
        self.optimizer = opt_func(model.parameters(), lr)
        self.print_interval = print_interval    
        self.lr_scheduler = lr_scheduler

        self.CURRENT_DIR = "/workspace"
        self.model_dir = os.path.join(self.CURRENT_DIR, model_dir)

        self.early_stopping = EarlyStopping(patience=patience, path=os.path.join(self.model_dir, "checkpoint.pth"))
        self.init_optimizer_scheduler(**scheduler_kwargs)
        self.log_to_wandb = log_to_wandb
        self.wandb_name = wandb_name


        self.config = dict(
            lr=self.lr,
            opt_func=opt_func.__name__,
            patience=patience,
            damping_factor=scheduler_kwargs.get('factor', None),
            linear_layer_sizes = [layer.out_features for layer in self.model.network if isinstance(layer, nn.Linear)],
            conv_layer_sizes = [layer.out_channels for layer in self.model.network if isinstance(layer, nn.Conv2d)],
        )

        if self.log_to_wandb:
            self.wrun = wandb.init(project="magnetic_dipole_inversion", entity="tm95mon", name=self.wandb_name, config=self.config)
            self.wrun.watch(model, log="all", log_freq=50) 
            self.model_artifact = wandb.Artifact(self.wandb_name, type="model")

    def init_optimizer_scheduler(self, **scheduler_kwargs):
        self.scheduler = self.lr_scheduler(
            self.optimizer, **scheduler_kwargs
        )
        return self
    
    def _train_on_batch(self, batch):
        loss = self.model.training_step(batch)
        loss.backward()
        self.optimizer.step()
        # log params and gradients to wandb
        if self.log_to_wandb:
            self.log_params_to_wandb(self.epoch)
        self.optimizer.zero_grad()
        return loss
    
    def fit(self, epochs, train_loader, val_loader):
        try:
            for epoch in range(epochs):
                self.epoch = epoch
                self.model.train()
                train_losses = []
                for batch in train_loader:
                    loss = self._train_on_batch(batch)
                    train_losses.append(loss)
                
                # Validation phase
                self.model.eval()
                result = self.model.evaluate(val_loader)
                result['train_loss'] = torch.stack(train_losses).mean().item()

                train_loss = torch.stack(train_losses).mean().item()
                val_loss = result['val_loss']
                val_acc = result['val_acc']

                # dynamic learning rate adjustment
                self.scheduler.step(val_loss)

                # log history
                self.log_history(train_loss, val_loss, val_acc)
                if self.log_to_wandb:
                    self.log_main_params_to_wandb(epoch, **self.history[-1])

                if (epoch) % self.print_interval == 0:
                    self.model.epoch_end(epoch, result)

                # early stopping 
                self.early_stopping(val_loss, self.model)

                if self.early_stopping.early_stop:
                    print("Early stopping")
                    # load the last checkpoint with the best model
                    self.model.load_state_dict(torch.load('checkpoint.pth'))
                    break
            if self.log_to_wandb:
                self.model_artifact.add_file(os.path.join(self.model_dir, "temp.pt"))
                self.wrun.log_artifact(self.model_artifact)
        finally:
            if self.log_to_wandb:
                self.wrun.finish()
        return self.history

    def log_history(self, train_loss, val_loss, val_acc):
        lr = self.optimizer.param_groups[0]['lr']
        self.history.append({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': lr
        })
    
    def log_params_to_wandb(self, epoch):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.wrun.log({f"{name}_grad_norm": param.grad.norm(),
                        f"{name}_grad_mean": param.grad.mean(),
                        f"{name}_grad_std": param.grad.std(),
                        f"{name}_grad_max": param.grad.max(),
                        f"{name}_grad_min": param.grad.min()}, step=epoch)
                self.wrun.log({f"{name}_param_norm": param.norm(),
                        f"{name}_param_mean": param.mean(),
                        f"{name}_param_std": param.std(),
                        f"{name}_param_max": param.max(),
                        f"{name}_param_min": param.min()}, step=epoch)
    
    def log_main_params_to_wandb(self, epoch, train_loss, val_loss, val_acc, lr):
        self.wrun.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': lr
            }, step=epoch)

        
