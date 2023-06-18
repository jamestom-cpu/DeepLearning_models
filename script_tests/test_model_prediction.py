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

from my_packages.neural_network.data_generators.mixed_array_generator import MixedArrayGenerator
from my_packages.neural_network.data_generators.iterator import DataIterator
from my_packages.neural_network.model.model_trainer import Trainer
from my_packages.neural_network.model.model_base import Model_Base
from my_packages.neural_network.predictor.predictor  import Predictor
from my_packages.neural_network.aux_funcs.evaluation_funcs import f1_score_np
# torch import 
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary


print("cuda available: ", torch.cuda.is_available())


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

# generate the data
random_fields, l = rmg.generate_labeled_data()
Hx, Hy, Ez = rmg.input_data_to_Scan(random_fields)

# inspect the generated data
fig, ax = plt.subplots(1,3, figsize=(15,5))
Hx.plot(ax=ax[0])
Hy.plot(ax=ax[1])
Ez.plot(ax=ax[2])
plt.show()


class CNN(Model_Base):
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
    
    # overwrite
    def print_summary(self, device = "cpu"):
        return summary(self, input_size=self.in_shape, device=device)




# predictor 
def preprocessing(x):
    H = x[-2:]
    maxH = np.max(np.abs(H))
    minH = np.min(np.abs(H))
    H = (H - minH)/(maxH - minH)
    return H

def postprocessing(x):
    return 1/(1+np.exp(-x)) # sigmoid

in_shape = (2,21,21); out_shape = (2,7,7)
conv_layer1_size = 32
conv_layer2_size = 64
linear_layer1_size = 256
model= CNN(
    in_shape=in_shape, out_shape=out_shape, 
    conv_size1=conv_layer1_size, conv_size2=conv_layer2_size, 
    linear_size1=linear_layer1_size)

model_dir = "models/simple_magnetic"
model_name = "temp.pt"

model_path = os.path.join(model_dir, model_name)

## load mlflow model
import mlflow.pytorch
mlflow_model_path = r"/workspace/mlflow/285319224310435874/320b40490b5445cb86d8ef2efde49834/artifacts/models"
mlflow_model = mlflow.pytorch.load_model(mlflow_model_path)


predictor = Predictor(
    preprocessing_func=preprocessing, 
    postprocessing_func=postprocessing,
    model=mlflow_model)
# predictor.load_model_and_weights(model_path, device="cuda")

prediction = predictor.predict(random_fields)
probability_map = predictor.prediction_probability_map(random_fields)



labelsH = l[-2:]
accuracy = predictor.accuracy(random_fields, labelsH, certainty_level=0.5)


fig, ax = plt.subplots(3,2, figsize=(15,4.5), constrained_layout=True)
rmg.plot_Hlabeled_data(random_fields, labelsH, ax=(ax[0,0], ax[0,1]))
predictor.plot(random_fields, certainty_level=0.5, ax= ax[1:])
plt.show()

print("finished")