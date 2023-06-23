import os, sys
import numpy as np
import scipy
import math as m
import cmath as cm
from pprint import pprint


# PROJECT_CWD = r"/home/pignari/Desktop/neural_networks/DeepLearning_models"
PROJECT_CWD = r"/workspace"
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
resolution=(11,11)
field_res = (30,30)
xbounds = [-1e-2, 1e-2]
ybounds = [-1e-2, 1e-2]
padding = None
dipole_height = 1e-3
substrate_thickness = 1.4e-2
substrate_epsilon_r = 4.4
dynamic_range = 2
probe_heights = [6e-3, 8e-3, 1e-2]
dipole_density_E = 0.1
dipole_density_H = 0.1
inclde_dipole_position_uncertainty = False
# data_dir = "/share/NN_data/high_res_with_noise"
data_dir = "/workspace/NN_data/11_res_"


rmg = MixedArrayGenerator(
    resolution=resolution,
    xbounds=xbounds,
    ybounds=ybounds,
    padding=padding,
    dipole_height=dipole_height,
    substrate_thickness=substrate_thickness,
    substrate_epsilon_r=substrate_epsilon_r,
    probe_height=probe_heights,
    dynamic_range=dynamic_range,
    f=[1e9],
    field_res=field_res,
    dipole_density_E=dipole_density_E,
    dipole_density_H=dipole_density_H,
    include_dipole_position_uncertainty=inclde_dipole_position_uncertainty,
    )


print("cell size is: {:.2f} x {:.2f} mm".format(rmg.cell_size[0]*1e3, rmg.cell_size[1]*1e3))
# dfh = rmg._generate_fh()
# M_magnetic = np.abs(dfh.magnetic_array.M)
# M_electric = np.abs(dfh.electric_array.M)

# magn_dyn_range = np.max(M_magnetic)/np.min(M_magnetic)
# electr_dyn_range = np.max(M_electric)/np.min(M_electric)

# print(magn_dyn_range, electr_dyn_range)


data_iterator = DataIterator(rmg)

N = 100000
N_test = 1000

inputs, target = data_iterator.generate_N_data_samples(N)
inputs_test, target_test = data_iterator.generate_N_data_samples(N_test)
train_and_valid_dataset = TensorDataset(torch.from_numpy(inputs).float(), torch.from_numpy(target).float())
test_dataset = TensorDataset(torch.from_numpy(inputs_test).float(), torch.from_numpy(target_test).float())
print("train_dataset size: ", len(train_and_valid_dataset))


# f, l = test_dataset[0]

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(len(probe_heights), 3, figsize=(15,5), constrained_layout=True)
# for ii in range(len(probe_heights)):
#     rmg.plot_labeled_data(f, l, savename="test_data", ax=ax[ii], index=ii)
# fig.suptitle("test data - probe heights = {}".format(probe_heights))
# plt.show()



if not os.path.exists(data_dir):
    os.makedirs(data_dir)
fullpath_train = os.path.join(data_dir, "train_and_valid_dataset.pt")
fullpath_test = os.path.join(data_dir, "test_dataset.pt")

torch.save(train_and_valid_dataset, fullpath_train)
torch.save(test_dataset, fullpath_test)
