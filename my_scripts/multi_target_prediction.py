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

#%% Inspect the Test Data

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

input, target = Hds_test[6]
probe_heights = itemgetter(*height_indices)(properties["probe_height"])

print("input shape: ", input.shape)
print("target shape: ", target.shape)
print("probe heights: ", probe_heights)
plotter.plot_fields_and_magnitude(input, target, index=0)
plt.show()

#%% Load The Model 
from my_packages.neural_network.model.multi_target_model import MultiTargetModel

input_shape = (4,30,30)
output_shape = (2,11,11)
binary_output_shape = (2,11,11)

print("loading model")
model = MultiTargetModel(input_shape, output_shape, binary_output_shape)
model = model.to(get_default_device())
summary(model, input_shape)

# load the model
model_dir = "models/multi_task"
model_name = "multi_task_model_2.pt"

print("load model state dictionary")
model_path = os.path.join(model_dir, model_name)
model.load_state_dict(torch.load(model_path))

print("model loaded")
#%% Define Predictor Class
class DipolePredictor:
    def __init__(self, model):
        self.model = model

    def _ensure_batch_size_of_one(self, input_data):
        if len(input_data.shape) == 3:
            input_data = input_data.reshape(1, *input_data.shape)
        return input_data

    def predict_binary(self, input_data):
        input_data = self._ensure_batch_size_of_one(input_data)
        # Reshaping the input data to (4, 30, 30)
        input_data = input_data.reshape(-1, 4, 30, 30)
        self.model.eval()
        with torch.no_grad():
            binary_prediction, _ = self.model(torch.Tensor(input_data).to(self.model.device))
            binary_prediction = torch.sigmoid(binary_prediction)
        return binary_prediction.cpu().numpy()

    def predict_magnitude_with_label(self, input_data, binary_labels):
        input_data = self._ensure_batch_size_of_one(input_data)
        binary_labels = self._ensure_batch_size_of_one(binary_labels)
        # Reshaping the input data to (4, 30, 30)
        input_data = input_data.reshape(-1, 4, 30, 30)
        self.model.eval()
        with torch.no_grad():
            _, magnitude_prediction = self.model(torch.Tensor(input_data).to(self.model.device), torch.Tensor(binary_labels).to(self.model.device))
        return magnitude_prediction.cpu().numpy()

    def predict_magnitude(self, input_data):
        input_data = self._ensure_batch_size_of_one(input_data)
        binary_predictions = self.predict_binary(input_data)
        magnitude_predictions = self.predict_magnitude_with_label(input_data, binary_predictions)
        return magnitude_predictions
    
#%% Test Predictions
    
predictor = DipolePredictor(model)
# # Assume input_data is your input with shape (2, 2, 30, 30)

input_data, target_data = Hds_test[0]

print("input shape: ", input_data.shape)
print("target shape: ", target_data.shape)
# plotter.plot_fields_and_magnitude(input_data, target_data, index=0)


#%% binary predictions
binary_predictions = predictor.predict_binary(input_data)[0]
print("binary predictions shape: ", binary_predictions.shape)
certainty = 0.7
binary_labels_prediction = (binary_predictions > certainty).astype(int)
binary_labels = target_data[0]

fig, ax = plt.subplots(2,2, figsize=(10,10), constrained_layout=True)
plotter.plot_Hlabeled_data(input_data, binary_labels, index=0, ax=ax[0])
plotter.plot_Hlabeled_data(input_data, binary_labels_prediction, index=0, ax=ax[1])

titles = ["x target", "y target", "x prediction", "y prediction"]
for ii, axi in enumerate(ax.flatten()):
    axi.set_title(titles[ii])
plt.show()
#%% magnitude predictions
magnitude_predictions = predictor.predict_magnitude(input_data)[0]
print("magnitude predictions shape: ", magnitude_predictions.shape)
fig, ax = plt.subplots(2,2, figsize=(10,10), constrained_layout=True)
plotter.plot_target_magnitude(target_data, ax=ax[0])
plotter.plot_target_magnitude(magnitude_predictions, ax=ax[1])

#%%
plt.show()


#%%

# binary_predictions = predictor.predict_binary(input_data)
# magnitude_predictions_with_label = predictor.predict_magnitude_with_label(input_data, binary_labels)
# magnitude_predictions = predictor.predict_magnitude(input_data)