import numpy as np
import pandas as pd
import scipy
import math as m
import cmath
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

import h5py
import os, sys

from mpl_toolkits.mplot3d import Axes3D
from pprint import pprint


PROJECT_CWD = r"/workspace/"
sys.path.append(PROJECT_CWD)



import torch

print("cuda available: ", torch.cuda.is_available())
print("number of GPUs: ",torch.cuda.device_count())
print("I am currently using device number: ", torch.cuda.current_device())
print("the device object is: ", torch.cuda.device(0))
print("the device name is: ", torch.cuda.get_device_name(0))


class DataIterator:
    def __init__(self, generating_class, normalize=True):
        self.gen = generating_class
        self.normalize = normalize

    def __iter__(self):
        return self

    def __next__(self):
        labeled_data = self.gen.generate_labeled_data()
        
        if self.normalize:
            labeled_data = self._normalize_data(labeled_data)
        
        return labeled_data

    def _normalize_data(self, labeled_data):
        f, t = labeled_data
        if np.sum(t) == 0:
            return labeled_data
        min_value = np.min(f)
        max_value = np.max(f)
        normalized_f = (f - min_value) / (max_value - min_value)
        return normalized_f, t
    
    def generate_N_data_samples(self, N):
        f = []
        t = []
        for _ in range(N):
            samplef, samplet = next(self)
            f.append(samplef)
            t.append(samplet)
        f = np.asarray(f)
        t = np.asarray(t)
        return f, t
    
from my_packages.neural_network.data_generators.magnetic_array_generator import RandomMagneticDipoleGenerator
from torch.utils.data import TensorDataset, DataLoader

resolution=(7,7)
field_res = (21,21)
xbounds = [-0.01, 0.01]
ybounds = [-0.01, 0.01]
dipole_height = 1e-3
substrate_thickness = 1.4e-2
substrate_epsilon_r = 4.4
dynamic_range = 10
probe_height = 0.3e-2
dipole_density = 0.2

rmg = RandomMagneticDipoleGenerator(
    resolution=resolution,
    xbounds=xbounds,
    ybounds=ybounds,
    dipole_height=dipole_height,
    substrate_thickness=substrate_thickness,
    substrate_epsilon_r=substrate_epsilon_r,
    dynamic_range=dynamic_range,
    probe_height=probe_height,
    field_res=field_res,
    dipole_density=dipole_density
    )


def fit(num_epochs, model, loss_fn, opt, train_dl, val_dl):
    loss_history = []  # List to store the loss values
    val_loss_history = []  # List to store the validation loss values
    
    for epoch in range(num_epochs):
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            
            loss.backward()
            opt.step()
            opt.zero_grad()
        
        # Append the loss value to the history list
        loss_history.append(loss.item())
        
        
        
        # Evaluate the model on the validation set
        with torch.no_grad():
            val_loss = 0.0
            for val_xb, val_yb in val_dl:
                val_pred = model(val_xb)
                val_loss += loss_fn(val_pred, val_yb).item()
            
            val_loss /= len(val_dl)
            val_loss_history.append(val_loss)
        
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item(), val_loss))
            
    return loss_history, val_loss_history
    

# create a dataset
N = 10000
batch_size = 5
learning_rate = 1e-3
n_iterations = 200

print("generate dataset")
data_iterator = DataIterator(rmg)
f, t = data_iterator.generate_N_data_samples(N)
f = f.reshape(N, -1)
t = t.reshape(N, -1)
f = torch.from_numpy(f).float()
t = torch.from_numpy(t).float()
print("dataset created")

# define the dataset
train_ds = TensorDataset(f, t)


# create validation set
val_fraction = 0.2  # Fraction of data to be used for validation
val_size = int(val_fraction * N)
test_fraction = 0.1  # Fraction of data to be used for testing
test_size = int(test_fraction * N)

# Split the dataset into training, validation, and test sets
train_data = TensorDataset(*train_ds[:N - val_size - test_size])
val_data = TensorDataset(*train_ds[N - val_size - test_size:N - test_size])
test_data = TensorDataset(*train_ds[N - test_size:])


# Create data loaders for training, validation, and test
train_dl = DataLoader(train_data, batch_size, shuffle=True)
val_dl = DataLoader(val_data, batch_size, shuffle=False)
test_dl = DataLoader(test_data, batch_size, shuffle=False)

# print the sizes of the training, validation, and test sets
print("dataset sizes: ", len(train_data), len(val_data), len(test_data))

# print the shapes of the input and target tensors
print("dataset shapes: ", train_ds.tensors[0].shape, train_ds.tensors[1].shape)

A = f.shape[1]
B = t.shape[1]
model = torch.nn.Linear(A, B)
loss_fn = torch.nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss_history, validation_loss_history = fit(n_iterations, model, loss_fn, opt, train_dl, val_dl)

# Evaluate the model on the test set
with torch.no_grad():
    test_loss = 0.0
    for test_xb, test_yb in test_dl:
        test_pred = model(test_xb)
        test_loss += loss_fn(test_pred, test_yb).item()
    
    test_loss /= len(test_dl)
    print('Test Loss: {:.4f}'.format(test_loss))

# plot loss and validation loss

fig, ax = plt.subplots()
ax.plot(loss_history, label='training loss')
ax.plot(validation_loss_history, label='validation loss')
plt.show()