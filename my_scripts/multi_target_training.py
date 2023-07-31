#%%
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

from my_packages.classes.aux_classes import Grid

from my_packages.neural_network.data_generators.mixed_array_generator import MixedArrayGenerator
from my_packages.neural_network.data_generators.array_generator_mag_and_phase import ArrayGenerator_MagnitudesAndPhases
from my_packages.neural_network.data_generators.iterator import DataIterator
from my_packages.neural_network.model.model_trainers.dipole_position_trainer import Trainer
from my_packages.neural_network.model.model_base import Model_Base
from my_packages.neural_network.plotting_functions.datapoints_plotting import Plotter

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

#%% create the data generators


data_dir = "/ext_data/NN_data/11_res_noise_MP_labels/"

# load the data properties
json_file = os.path.join(data_dir, "data_properties.json")
with open(json_file, "r") as f:
    properties = json.load(f)

properties["probe_height"] = [6e-3, 8e-3, 1e-2, 1.2e-2]

fullpath_train = os.path.join(data_dir, "training.pt")
fullpath_test = os.path.join(data_dir, "test.pt")


# load the data from the datasets
height_indices = [0, -1]
Hds = H_Components_Dataset_Multilayer(fullpath_train, height_indices=height_indices)
Hds = Hds.rescale_probe_heights().rescale_labels()
Hds_test = H_Components_Dataset_Multilayer(fullpath_test, height_indices=height_indices)
Hds_test = Hds_test.rescale_probe_heights().rescale_labels()


#%% inspect the dataset
from operator import itemgetter

# load the data properties
json_file = os.path.join(data_dir, "data_properties.json")
with open(json_file, "r") as f:
    properties = json.load(f)



xaxis = np.linspace(*properties["xbounds"], properties["field_res"][0])
yaxis = np.linspace(*properties["ybounds"], properties["field_res"][1])
z = np.array(properties["probe_height"])

grid = Grid(np.meshgrid(xaxis, yaxis, z, indexing="ij"))
plotter = Plotter.initialize_from_res(
    grid, properties["resolution"], properties["dipole_height"], 1e9
)

input, target = Hds[6]
probe_heights = itemgetter(*height_indices)(properties["probe_height"])

print("input shape: ", input.shape)
print("target shape: ", target.shape)
print("probe heights: ", probe_heights)
plotter.plot_fields_and_magnitude(input, target, index=0)
plt.show()



# %% Define Model Type
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from my_packages.neural_network.model.model_base import Model_Base
from NN_model_architectures.NN_blocks import simple_conv_block, conv_block, linear_block

from torchsummary import summary
from torchviz import make_dot


from my_packages.neural_network.model.model_base import Model_Base
from NN_model_architectures.NN_blocks import simple_conv_block, conv_block, linear_block
from NN_model_architectures.PredictDipoleMoments.MultiTask1 import Convolutional_Base, BinaryPredictionHead, DipoleMagnitudePredictionHead
from my_packages.neural_network.model.multi_target_model import MultiTargetModel
    
    
input_shape = (4,30,30)
output_shape = (2,11,11)
binary_output_shape = (2,11,11)

model = MultiTargetModel(input_shape, output_shape, binary_output_shape)
onnx_path = "onnx_models/myModel.onnx"
model.export_to_onnx(onnx_path)

to_device(model, torch.device('cuda'))
summary(model, input_shape)
#%% Define Training


from my_packages.neural_network.model.model_trainers.magnitude_multitask_trainer import Trainer

    
#%% Define DataLoaders
# create the dataloaders
train_size = int(0.8 * len(Hds))
val_size = len(Hds) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(Hds.view_with_shape(input_shape), [train_size, val_size])

train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_dataset, batch_size=32, num_workers=4, pin_memory=True)

#%% Train the model
# params
loss_fn_binary = nn.BCEWithLogitsLoss()
loss_fn_magnitude = nn.MSELoss()
# model dir
model_dir = os.path.join(PROJECT_CWD, "models", "simple_electric")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
experiment_name = "MultiTask0_dipole_position"
run_name = "largeRN"

# training parameters
lr = 0.001
patience = 5
lr_dampling_factor = 0.5
lr_patience = 0
opt_func = torch.optim.Adam
n_iterations = 5
# regularization
weight_decay= 1e-6

trainer = Trainer(
    model, opt_func=opt_func,
    loss_fn_binary=loss_fn_binary, loss_fn_magnitude=loss_fn_magnitude,
    lr=lr, patience=patience,
    optimizer_kwargs={"weight_decay":weight_decay},
    scheduler_kwargs={'mode':'min', 'factor':lr_dampling_factor, 'patience':lr_patience,
                      'verbose':True},
    model_dir=model_dir, experiment_name=experiment_name, run_name=run_name,
    log_gradient=["conv1", "conv2", "fc1"], log_weights=[], parameters_of_interest={}, 
    print_every_n_epochs=1,
    log_mlflow=False, log_tensorboard=False
    )

device = get_default_device()
model = to_device(model, device)

# move loaders to device (GPU)
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)

print("evaluation before training: ", model.evaluate_binary(val_dl, loss_fn_binary))
print("evaluation before training: ", model.evaluate_magnitude(val_dl, loss_fn_magnitude))

trainer.lr = 0.001
trainer.fit_binary(20, train_dl, val_dl)
trainer.switch_to_magnitude()
trainer.fit_magnitude(20, train_dl, val_dl)
trainer.lr = 0.0005
trainer.reset_model_training()
trainer.fit_binary(20, train_dl, val_dl)
trainer.switch_to_magnitude()
trainer.fit_magnitude(20, train_dl, val_dl)
trainer.lr = 0.0001
trainer.reset_model_training()
trainer.fit_magnitude(20, train_dl, val_dl)

# save the model

model_dir = "models/multi_task"
model_name = "multi_task_model_3.pt"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, model_name)
torch.save(model.state_dict(), model_path)


# #%% Evaluate the model

# class DipolePredictor:
#     def __init__(self, model):
#         self.model = model

#     def predict_binary(self, input_data):
#         # Reshaping the input data to (4, 30, 30)
#         input_data = input_data.reshape(-1, 4, 30, 30)
#         self.model.eval()
#         with torch.no_grad():
#             binary_prediction, _ = self.model(torch.Tensor(input_data).to(self.model.device))
#             binary_prediction = torch.sigmoid(binary_prediction)
#         return binary_prediction.cpu().numpy()

#     def predict_magnitude_with_label(self, input_data, binary_labels):
#         # Reshaping the input data to (4, 30, 30)
#         input_data = input_data.reshape(-1, 4, 30, 30)
#         self.model.eval()
#         with torch.no_grad():
#             _, magnitude_prediction = self.model(torch.Tensor(input_data).to(self.model.device), torch.Tensor(binary_labels).to(self.model.device))
#         return magnitude_prediction.cpu().numpy()

#     def predict_magnitude(self, input_data):
#         binary_predictions = self.predict_binary(input_data)
#         magnitude_predictions = self.predict_magnitude_with_label(input_data, binary_predictions)
#         return magnitude_predictions
    

# predictor = DipolePredictor(model)
# # Assume input_data is your input with shape (2, 2, 30, 30)
# binary_predictions = predictor.predict_binary(input_data)
# magnitude_predictions_with_label = predictor.predict_magnitude_with_label(input_data, binary_labels)
# magnitude_predictions = predictor.predict_magnitude(input_data)