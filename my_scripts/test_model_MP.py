print("starting script")


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
import json



PROJECT_CWD = r"/workspace/"
sys.path.append(PROJECT_CWD)

os.chdir(PROJECT_CWD)
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
os.environ["MLFLOW_TRACKING_URI"] = "mlflow"

from my_packages.neural_network.data_generators.mixed_array_generator import MixedArrayGenerator
from my_packages.neural_network.data_generators.array_generator_mag_and_phase import ArrayGenerator_MagnitudesAndPhases
from my_packages.neural_network.data_generators.iterator import DataIterator
from my_packages.neural_network.model.model_trainers.dipole_position_trainer import Trainer
from my_packages.neural_network.model.model_base import Model_Base

# torch import
import torch
from torch.utils.data import TensorDataset, DataLoader

print("cuda available: ", torch.cuda.is_available())
print("number of GPUs: ",torch.cuda.device_count())
print("I am currently using device number: ", torch.cuda.current_device())
print("the device object is: ", torch.cuda.device(0))
print("the device name is: ", torch.cuda.get_device_name(0))
torch.cuda.empty_cache()


from my_packages.neural_network.model.early_stopping import EarlyStopping


# consider the GPU
from my_packages.neural_network.gpu_aux import get_default_device, to_device, DeviceDataLoader
from torchsummary import summary

from torch import nn
import torch.nn.functional as F
from my_packages.neural_network.datasets_and_loaders.dataset_transformers_multilayer import H_Components_Dataset_Multilayer
from my_packages.neural_network.datasets_and_loaders.dataset_transformers_E import E_Components_Dataset

from singleton_python_objects.mixed_array_generator import get_mixed_array_generator
from NN_model_architectures.PredictDipolePosition.ResNet import get_model



# data_dir = "/share/NN_data/high_res_with_noise"
# data_dir = "/ext_data/NN_data/11_res_noise/"
data_dir = "/ext_data/NN_data/11_res_noise_MP_labels/"

# load the data properties
json_file = os.path.join(data_dir, "data_properties.json")
with open(json_file, "r") as f:
    properties = json.load(f)

properties["probe_height"] = [6e-3, 8e-3, 1e-2, 1.2e-2]


# rmg = MixedArrayGenerator(**properties)
rmg_dp = ArrayGenerator_MagnitudesAndPhases(
    **properties,
    )

# rmg = get_mixed_array_generator()
data_iterator = DataIterator(rmg_dp)


fields,labels = data_iterator.generate_N_data_samples(10)

f, t = fields[0], labels[0]

height_index = 0

ds = TensorDataset(torch.from_numpy(fields), torch.from_numpy(labels))
# Eds = E_Components_Dataset(ds, probe_height_index=height_index).scale_to_01()
Hds = H_Components_Dataset_Multilayer(ds).rescale_probe_heights().rescale_labels().expand_phase()

fH, lH = Hds[0]

# rmg_dp.plot_target_magnitude(lH)
# rmg_dp.plot_target_phase(lH, normalized01=True)
rmg_dp.plot(fH, lH, index=height_index)
plt.show()
# rmg_dp.plot(f, t, index=height_index)
# plt.show()



# N = 100000
# N_test = 1000


# # save the datasets
# save_dir = os.path.join(PROJECT_CWD, "NN_data", "mixed_array_data")
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# fullpath_train = os.path.join(save_dir, "train_and_valid_dataset.pt")
# fullpath_test = os.path.join(save_dir, "test_dataset.pt")


fullpath_train = os.path.join(data_dir, "training.pt")
fullpath_test = os.path.join(data_dir, "test.pt")


# load the data from the datasets
train_and_valid_dataset = torch.load(fullpath_train)
test_dataset = torch.load(fullpath_test)

Hds = H_Components_Dataset_Multilayer(train_and_valid_dataset).rescale_probe_heights().rescale_labels()



# split into training and validation sets
train_size = int(0.8 * len(Hds))
val_size = len(Hds) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(Hds, [train_size, val_size])

print("train_dataset size: ", len(train_dataset))
print("val_dataset size: ", len(val_dataset))


# set the device (cuda or cpu)
device = get_default_device()
print("device: ", device)

# inspect data
n_examples = 5


# plot the examples
plt.switch_backend('TkAgg')
fig, axs = plt.subplots(n_examples, 2, figsize=(15, 5))

for i in range(n_examples):
    H_examples, labels_examples = train_dataset[i]
    rmg_dp.plot(H_examples, labels_examples, ax=axs[i])
plt.show()


## create the data loaders

batch_size = 256

# create the dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True ,num_workers = 4,  pin_memory=True)
test_dataloader = DataLoader(Hds_test, batch_size=batch_size, shuffle=True)
# move the dataloaders to the GPU
train_dl = DeviceDataLoader(train_dataloader, device)
val_dl = DeviceDataLoader(val_dataloader, device)

## build the model
input_shape =   (4, 30, 30)
output_shape =  (6, 11, 11)
loss_fn = nn.BCEWithLogitsLoss()

model = get_model(input_shape, output_shape)
print(model.print_summary(device="cpu"))


# model dir
model_dir = os.path.join(PROJECT_CWD, "models", "simple_electric")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
experiment_name = "test"
run_name = "largeRN"

## Train the Model
# training parameters
lr = 0.001
patience = 2
lr_dampling_factor = 0.5
lr_patience = 0
opt_func = torch.optim.Adam
n_iterations = 100
# regularization
weight_decay= 1e-6

trainer = Trainer(
    model, opt_func=opt_func,
    lr=lr, patience=patience,
    optimizer_kwargs={"weight_decay":weight_decay},
    scheduler_kwargs={'mode':'min', 'factor':lr_dampling_factor, 'patience':lr_patience,
                      'verbose':True},
    model_dir=model_dir, experiment_name=experiment_name, run_name=run_name,
    log_gradient=["conv1", "conv2", "fc1"], log_weights=[], parameters_of_interest={}, 
    print_every_n_epochs=1,
    log_mlflow=True, log_tensorboard=False
    )


model = to_device(model, device)
print("evaluation before training: ", model.evaluate(val_dl))


torch.cuda.empty_cache()


print("starting training")
history = trainer.fit(n_iterations, train_dl, val_dl)


# use the model to evaluate the test set
print("evaluation after training")
test_dl = DeviceDataLoader(test_dataloader, device)
print("evaluation on the test set: ", model.evaluate(test_dl))

# try clearing the cache
torch.save(model.state_dict(), os.path.join(model_dir, "temp.pt"))