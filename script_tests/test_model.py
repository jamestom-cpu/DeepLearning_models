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

os.chdir(PROJECT_CWD)
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
os.environ["MLFLOW_TRACKING_URI"] = "mlflow"

from my_packages.neural_network.data_generators.mixed_array_generator import MixedArrayGenerator
from my_packages.neural_network.data_generators.iterator import DataIterator
from my_packages.neural_network.model.model_trainer import Trainer
from my_packages.neural_network.model.CNN_base import CNN_Base

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




# split into training and validation sets
train_size = int(0.8 * len(Hds))
val_size = len(Hds) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(Hds, [train_size, val_size])

print("train_dataset size: ", len(train_dataset))
print("val_dataset size: ", len(val_dataset))

## Define the Model Structure
class CNN(CNN_Base):
    def __init__(self, in_shape, out_shape, conv_size1=32, conv_size2=64, linear_size1 = 128, loss_fn=F.mse_loss):
        
        self.in_shape = in_shape
        self.out_shape = out_shape
        
        n_layers = self.in_shape[0]
        out_size = np.prod(out_shape)
        super(CNN, self).__init__(loss_fn=loss_fn)

        # conv layers
        self.conv1 = nn.Conv2d(n_layers, conv_size1, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(conv_size1, conv_size2, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        # model head
        self.maxpool = nn.MaxPool2d(2, 2) # output: conv_size2 x 10 x 10
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(conv_size2 * 10 * 10, linear_size1)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(linear_size1, out_size)
        self.unflatten = nn.Unflatten(1, out_shape)

    
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)
        out = self.unflatten(out)
        return out
    
    def predict(self, inputs: np.ndarray):
        if isinstance(inputs, np.ndarray):
            if inputs.ndim == 3:
                inputs = np.expand_dims(inputs, axis=0)
            inputs = torch.from_numpy(inputs).float()
        return self(inputs).detach().numpy()
    
    
    def print_summary(self, device = "cpu"):
        return summary(self, input_size=self.in_shape, device=device)




device = get_default_device()
print("device: ", device)


batch_size = 64   
conv_layer1_size = 32
conv_layer2_size = 64
linear_layer1_size = 256
loss_fn = nn.BCEWithLogitsLoss()
lr = 0.001
patience = 5
lr_dampling_factor = 0.5
opt_func = torch.optim.Adam
n_iterations = 5


model_dir = os.path.join(PROJECT_CWD, "models", "test")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)




# create the dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True ,num_workers = 4,  pin_memory=True)
test_dataloader = DataLoader(Hds_test, batch_size=batch_size, shuffle=True)
# move the dataloaders to the GPU
train_dl = DeviceDataLoader(train_dataloader, device)
val_dl = DeviceDataLoader(val_dataloader, device)
test_dl = DeviceDataLoader(test_dataloader, device)

input_shape =   (2, 21, 21)
output_shape =  (2, 7, 7)

model = CNN(
    input_shape,
    output_shape, 
    conv_size1=conv_layer1_size, 
    conv_size2=conv_layer2_size, 
    linear_size1=linear_layer1_size,
    loss_fn=loss_fn)
print(model.print_summary(device="cpu"))

experiment_name = "test"
run_name = "separate_logging"

trainer = Trainer(
    model, opt_func=opt_func,
    lr=lr, patience=patience, 
    scheduler_kwargs={'mode':'min', 'factor':lr_dampling_factor, 'patience':2, 
                      'verbose':True}, 
    model_dir=model_dir, experiment_name=experiment_name, run_name=run_name,
    log_gradient=["conv1"], log_weights=[], parameters_of_interest={
        "conv_layer1_size": conv_layer1_size,
        "conv_layer2_size": conv_layer2_size,
    }
    )


model = to_device(model, device)
model.evaluate(val_dl)


# model dir
model_dir = os.path.join(PROJECT_CWD, "models", "simple_magnetic")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


history = trainer.fit(n_iterations, train_dl, val_dl)
# temp save of the model
# torch.save(model.state_dict(), os.path.join(model_dir, "temp.pt"))