import os, sys
import numpy as np
import scipy
import math as m
import cmath as cm
from pprint import pprint


PROJECT_CWD = r"/home/pignari/Desktop/neural_networks/DeepLearning_models"
sys.path.append(PROJECT_CWD)

from my_packages.neural_network.data_generators.mixed_array_generator import MixedArrayGenerator
from my_packages.neural_network.data_generators.iterator import DataIterator

from my_packages.neural_network.data_generators.mixed_array_generator import MixedArrayGenerator
from my_packages.neural_network.data_generators.iterator import DataIterator

# torch import 
import torch
from torch.utils.data import TensorDataset, DataLoader


print("cuda available: ", torch.cuda.is_available())
print("number of GPUs: ",torch.cuda.device_count())
if torch.cuda.is_available():
    print("I am currently using device number: ", torch.cuda.current_device())
    print("the device object is: ", torch.cuda.device(0))
    print("the device name is: ", torch.cuda.get_device_name(0))



# data parameters
resolution=(21,21)
field_res = (50,50)
xbounds = [-0.01, 0.01]
ybounds = [-0.01, 0.01]
dipole_height = 1e-3
substrate_thickness = 1.4e-2
substrate_epsilon_r = 4.4
dynamic_range = 2
probe_height = 0.6e-2
dipole_density_E = 0.05
dipole_density_H = 0.05
data_dir = "/share/NN_data/high_res_with_noise"


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

# dfh = rmg._generate_fh()
# M_magnetic = np.abs(dfh.magnetic_array.M)
# M_electric = np.abs(dfh.electric_array.M)

# magn_dyn_range = np.max(M_magnetic)/np.min(M_magnetic)
# electr_dyn_range = np.max(M_electric)/np.min(M_electric)

# print(magn_dyn_range, electr_dyn_range)


data_iterator = DataIterator(rmg)

N = 10
N_test = 5

inputs, target = data_iterator.generate_N_data_samples(N)
inputs_test, target_test = data_iterator.generate_N_data_samples(N_test)
train_and_valid_dataset = TensorDataset(torch.from_numpy(inputs).float(), torch.from_numpy(target).float())
test_dataset = TensorDataset(torch.from_numpy(inputs_test).float(), torch.from_numpy(target_test).float())
print("train_dataset size: ", len(train_and_valid_dataset))



if not os.path.exists(data_dir):
    os.makedirs(data_dir)
fullpath_train = os.path.join(data_dir, "train_and_valid_dataset.pt")
fullpath_test = os.path.join(data_dir, "test_dataset.pt")

torch.save(train_and_valid_dataset, fullpath_train)
torch.save(test_dataset, fullpath_test)
