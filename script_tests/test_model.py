import wandb
import os, sys
import h5py
import numpy as np
import pandas as pd
import scipy
import math as m
import cmath
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from pprint import pprint
plt.switch_backend('TkAgg')


PROJECT_CWD = r"/workspace/"
sys.path.append(PROJECT_CWD)

from my_packages.neural_network.data_generators.mixed_array_generator import MixedArrayGenerator
from my_packages.neural_network.data_generators.iterator import DataIterator

# torch import 
import torch
from torch.utils.data import TensorDataset, DataLoader

print("cuda available: ", torch.cuda.is_available())
print("number of GPUs: ",torch.cuda.device_count())
print("I am currently using device number: ", torch.cuda.current_device())
print("the device object is: ", torch.cuda.device(0))
print("the device name is: ", torch.cuda.get_device_name(0))



from my_packages.neural_network.model.early_stopping import EarlyStopping


# consider the GPU
from my_packages.neural_network.gpu_aux import get_default_device, to_device, DeviceDataLoader
from torchsummary import summary

from torch import nn
import torch.nn.functional as F
from my_packages.neural_network.datasets_and_loaders.dataset_transformers_H import H_Components_Dataset
from my_packages.neural_network.datasets_and_loaders.dataset_transformers_E import E_Components_Dataset

# data parameters
resolution=(7,7)
field_res = (21,21)
xbounds = [-0.01, 0.01]
ybounds = [-0.01, 0.01]
dipole_height = 1e-3
substrate_thickness = 1.4e-2
substrate_epsilon_r = 4.4
dynamic_range = 10
probe_height = 0.3e-2
dipole_density_E = 0.2
dipole_density_H = 0.2


rmg = MixedArrayGenerator(
    resolution=resolution,
    xbounds=xbounds,
    ybounds=ybounds,
    dipole_height=dipole_height,
    substrate_thickness=substrate_thickness,
    substrate_epsilon_r=substrate_epsilon_r,
    probe_height=probe_height,
    dynamic_range=dynamic_range,
    f=[1e9],
    field_res=field_res,
    dipole_density_E=dipole_density_E,
    dipole_density_H=dipole_density_H
    )

data_iterator = DataIterator(rmg)


N = 100000
N_test = 1000


# save the datasets
save_dir = os.path.join(PROJECT_CWD, "NN_data", "mixed_array_data")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
fullpath_train = os.path.join(save_dir, "train_and_valid_dataset.pt")
fullpath_test = os.path.join(save_dir, "test_dataset.pt")

# load the data from the datasets
train_and_valid_dataset = torch.load(fullpath_train)
test_dataset = torch.load(fullpath_test)



Hds = H_Components_Dataset(train_and_valid_dataset).scale_to_01()
Eds = E_Components_Dataset(train_and_valid_dataset).scale_to_01()

Hds_test = H_Components_Dataset(test_dataset).scale_to_01() 
Eds_test = E_Components_Dataset(test_dataset).scale_to_01()


batch_size = 64

# split into training and validation sets
train_size = int(0.8 * len(Hds))
val_size = len(Hds) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(Hds, [train_size, val_size])

print("train_dataset size: ", len(train_dataset))
print("val_dataset size: ", len(val_dataset))

# create the dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True ,num_workers = 4,  pin_memory=True)
test_dataloader = DataLoader(Hds_test, batch_size=batch_size, shuffle=True)

class CNN_Base(nn.Module):
    def __init__(self, loss_fn=F.mse_loss, *args, **kwargs):
        super(CNN_Base, self).__init__()
        self.loss_fn = loss_fn


    def training_step(self, batch):
        inputs, targets = batch
        out = self(inputs)
        loss = self.loss_fn(out, targets)
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        out = self(inputs)
        loss = self.loss_fn(out, targets)
        accuracy = self._accuracy(out, targets)
        return {'val_loss': loss.detach(), 'val_acc': accuracy}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, 
            result['train_loss'],  
            result['val_loss'], 
            result['val_acc']))
    def evaluate(self, val_loader):
        self.eval() # set to evaluation mode
        outputs = [self.validation_step(batch) for batch in val_loader]
        return self.validation_epoch_end(outputs)
    
    @staticmethod
    def _accuracy(out, targets, thresh=0.5):
        with torch.no_grad():
            # Convert output probabilities to binary values (0 or 1)
            out_binary = (out > thresh).float()

            # Calculate true positives, false positives and false negatives
            true_positives = (out_binary * targets).sum().item()
            false_positives = (out_binary * (1 - targets)).sum().item()
            false_negatives = ((1 - out_binary) * targets).sum().item()

            # Calculate precision and recall
            precision = true_positives / (true_positives + false_positives + 1e-8)
            recall = true_positives / (true_positives + false_negatives + 1e-8)

            # Calculate F1 score
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
            return torch.tensor(f1_score)

class CNN(CNN_Base):
    def __init__(self, out_shape, conv_size1=32, conv_size2=64, linear_size1 = 128, loss_fn=F.mse_loss, n_layers=3):
        out_size = np.prod(out_shape)
        super(CNN, self).__init__(loss_fn=loss_fn)
        self.network = nn.Sequential(
            nn.Conv2d(n_layers, conv_size1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_size1, conv_size2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: conv_size2 x 10 x 10

            nn.Flatten(),
            nn.Linear(conv_size2 * 10 * 10, linear_size1),
            nn.ReLU(),
            nn.Linear(linear_size1, out_size),
            nn.Unflatten(1, out_shape)

        )
    
    def forward(self, xb):
        return self.network(xb)




input_shape =   (2, 21, 21)
output_shape =  (2, 7, 7)
out_size = 7*7*3

model = CNN(output_shape, n_layers=2)

summary(model, input_shape, batch_size=-1, device='cpu')

device = get_default_device()
print("device: ", device)

# move the dataloaders to the GPU
train_dl = DeviceDataLoader(train_dataloader, device)
val_dl = DeviceDataLoader(val_dataloader, device)
test_dl = DeviceDataLoader(test_dataloader, device)


class Trainer:
    def __init__(self, model, opt_func=torch.optim.SGD, lr=0.01, patience=7, scheduler_kwargs={}):
        self.model = model
        self.history = []
        self.lr = lr
        self.optimizer = opt_func(model.parameters(), lr)
        self.early_stopping = EarlyStopping(patience=patience, path='checkpoint.pth')
        self.init_optimizer_scheduler(**scheduler_kwargs)
    
    def init_optimizer_scheduler(self, **scheduler_kwargs):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, **scheduler_kwargs
            )
        return self
    
    def _train_on_batch(self, batch):
        loss = self.model.training_step(batch)
        loss.backward()
        self.optimizer.step()
        # log params and gradients to wandb
        self.log_params_to_wandb(self.epoch)
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
            self.log_main_params_to_wandb(epoch, **self.history[-1]) 

             

            if (epoch) % 5 == 0:
                self.model.epoch_end(epoch, result)

            # early stopping 
            self.early_stopping(val_loss, self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                # load the last checkpoint with the best model
                self.model.load_state_dict(torch.load('checkpoint.pth'))
                break
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
    
    def log_params_to_wandb(self, epoch):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                wandb.log({f"{name}_grad_norm": param.grad.norm(),
                        f"{name}_grad_mean": param.grad.mean(),
                        f"{name}_grad_std": param.grad.std(),
                        f"{name}_grad_max": param.grad.max(),
                        f"{name}_grad_min": param.grad.min()}, step=epoch)
                wandb.log({f"{name}_param_norm": param.norm(),
                        f"{name}_param_mean": param.mean(),
                        f"{name}_param_std": param.std(),
                        f"{name}_param_max": param.max(),
                        f"{name}_param_min": param.min()}, step=epoch)
    def log_main_params_to_wandb(self, epoch, train_loss, val_loss, val_acc, lr):
        # Log accuracy and loss to wandb
        wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': lr
            }, step=epoch)
        

output_shape = (2, 7, 7)
conv_layer1_size = 32
conv_layer2_size = 64
linear_layer1_size = 128
loss_fn = nn.BCEWithLogitsLoss()
lr = 0.001
patience = 5
lr_dampling_factor = 0.5
opt_func = torch.optim.Adam

model = CNN(
    output_shape, 
    n_layers=2,
    conv_size1=conv_layer1_size, 
    conv_size2=conv_layer2_size, 
    linear_size1=linear_layer1_size,
    loss_fn=loss_fn)

trainer = Trainer(
    model, opt_func=opt_func,
    lr=lr, patience=patience, 
    scheduler_kwargs={'mode':'min', 'factor':lr_dampling_factor, 'patience':2, 
                      'verbose':True})


model = to_device(model, device)

model.evaluate(val_dl)


# create a dataset
n_iterations = 50

# model dir
model_dir = os.path.join(PROJECT_CWD, "models", "simple_magnetic")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 2a6ac6cf8c981bd8a56480328e6affc93f086c07

config = dict(
    epochs=n_iterations,
    learning_rate=lr,
    conv_layer1_size=conv_layer1_size,
    conv_layer2_size=conv_layer2_size,
    linear_layer1_size=linear_layer1_size,
    loss_fn=loss_fn.__class__.__name__,
    opt_func=opt_func.__name__,
    dataset_size=N,
    test_dataset_size=N_test,
    batch_size=batch_size)

# increase timeout to 300s
os.environ["WANDB__SERVICE_WAIT"] = "300"

wrun = wandb.init(project="magnetic_dipole_inversion", entity="tm95mon", name="CNN2", config=config)
wandb.watch(model, log="all", log_freq=50) 

# save model 
model_artifact = wandb.Artifact('CNN_earlystopping_lr', type='model', description='simple 21x21 -> 7x7 mangetic model only', metadata=dict(config))

history = trainer.fit(n_iterations, train_dl, val_dl)
# temp save of the model
torch.save(model.state_dict(), os.path.join(model_dir, "temp.pt"))