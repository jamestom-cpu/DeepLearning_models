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
from types import SimpleNamespace
from typing import Tuple, List, Union, Iterable


plt.switch_backend('TkAgg')
PROJECT_CWD = r"/workspace/"
sys.path.append(PROJECT_CWD)

os.chdir(PROJECT_CWD)

# neuratl network imports
from my_packages.neural_network.data_generators.mixed_array_generator import MixedArrayGenerator
from my_packages.neural_network.data_generators.iterator import DataIterator
from my_packages.neural_network.model.model_trainer import Trainer
from my_packages.neural_network.model.model_base import Model_Base
from my_packages.neural_network.predictor.predictor  import Predictor
from my_packages.neural_network.aux_funcs.evaluation_funcs import f1_score_np

# hdf5 imports
from my_packages.hdf5.util_classes import Measurement_Handler
from my_packages.classes.field_classes import Scan
from my_packages.classes.aux_classes import Grid
from my_packages.field_databases.load_fields import FieldLoader


# torch import 
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary


print("cuda available: ", torch.cuda.is_available())

## inspect the data

# import the data generator 
from singleton_python_objects import Quasi_ResNet



# load the data
db_directory = "/ext_data/simulated_fields"
# db_name = "field_testing_for_susceptibility.hdf5"
db_name = "fields_for_testing.h5"
# savename = "RSTEST_2_LINEs.hdf5"
savename = "strip_with_patch"

field_loader = FieldLoader(db_directory, db_name, savename, probe_height=6e-3)

Hx, Hy, Ez = field_loader.Hx, field_loader.Hy, field_loader.Ez




fig, ax = plt.subplots(1,2, figsize=(10,5), constrained_layout=True)
Hx.plot_fieldmap(ax=ax[0])
Hy.plot_fieldmap(ax=ax[1])
plt.show()

# preprocess the data to obtain a smooth fieldmap
from my_packages.neural_network.preprocessing.fieldmaps import FieldmapPreprocessor

Hx_processed = FieldmapPreprocessor(Hx.scan).preprocess(sigma=0.1)
Hy_processed = FieldmapPreprocessor(Hy.scan).preprocess(sigma=0.1)

Hx_processed = Hx.update_with_scan(Hx_processed)
Hy_processed = Hy.update_with_scan(Hy_processed)    

fig, ax = plt.subplots(2,2, figsize=(10,5), constrained_layout=True)
Hx.plot_fieldmap(ax=ax[0,0])
Hy.plot_fieldmap(ax=ax[1,0])

Hx_processed.plot_fieldmap(ax=ax[0,1])
Hy_processed.plot_fieldmap(ax=ax[1,1])

fig, ax = plt.subplots(2,2, figsize=(10,5), constrained_layout=True)
Hx.plot(ax=ax[0,0])
Hy.plot(ax=ax[1,0])

Hx_processed.plot(ax=ax[0,1])
Hy_processed.plot(ax=ax[1,1])

plt.show()

## Extract patches from the fieldmaps
from my_packages.neural_network.large_scans.patch_extractor import ScanPatchExtractor
        
        

fill_value_for_padding = 0.0
patch_xbounds = (-2e-2, 2e-2)
patch_ybounds = (-2e-2, 2e-2)
patch_extractor_Hx = ScanPatchExtractor(
    Hx_processed, patch_xbounds=patch_xbounds, patch_ybounds=patch_ybounds,
    patch_shape=(30,30), padding_fill_value=fill_value_for_padding)
patch_extractor_Hy = ScanPatchExtractor(
    Hy_processed, patch_xbounds=patch_xbounds, patch_ybounds=patch_ybounds,
    patch_shape=(30,30), padding_fill_value=fill_value_for_padding)

# normalize 
vmax = np.max([Hx_processed.scan.max(), Hy_processed.scan.max()])
vmin = fill_value_for_padding

# normalize the batch extractors
patch_extractor_Hx.normalize(vmax=vmax, vmin=vmin)
patch_extractor_Hy.normalize(vmax=vmax, vmin=vmin)

# Hx_patch_center_raw = patch_extractor_Hx.get_simple_patch(
#     xbounds=patch_xbounds, ybounds=patch_ybounds
#     )
# Hx_patch_center = patch_extractor_Hx.get_patch(xbounds=patch_xbounds, ybounds=patch_ybounds
#     )

Hx_patches = patch_extractor_Hx.get_patches()
Hy_patches = patch_extractor_Hy.get_patches()

# ## compare a raw patch with a resamples patch
# fig, ax = plt.subplots(1,2, figsize=(10,5), constrained_layout=True)
# Hx_patch_center_raw.plot_fieldmap(ax=ax[0])
# Hx_patch_center.plot_fieldmap(ax=ax[1])
# ax[0].set_title("raw patch - Hx")
# ax[1].set_title("resampled patch - Hx")

## plot all patches
patch_extractor_Hx.plot_patches(Hx_patches, use_01_minmax=True)
patch_extractor_Hy.plot_patches(Hy_patches, use_01_minmax=True)

fig, ax = plt.subplots(figsize=(10,2), constrained_layout=True)
Hx_patches[1,0].plot_fieldmap(ax=ax)
plt.show()

# create a class that takes in patch extractors and returns as data for the neural network

def get_data(Hx_patches, Hy_patches):
    # create the data
    Hx_patches = np.array([patch.scan for patch in Hx_patches.flatten()], dtype=np.float32)
    Hy_patches = np.array([patch.scan for patch in Hy_patches.flatten()], dtype=np.float32)

    data = np.stack((Hx_patches, Hy_patches), axis=1)
    return data

data = get_data(Hx_patches, Hy_patches)

# plot all patches 

# predictor 
def preprocessing(x):
    # get the H fields
    return x

def postprocessing(x):
    return 1/(1+np.exp(-x)) # sigmoid


input_shape =   (2, 30, 30)
output_shape =  (2, 11, 11)

model = Quasi_ResNet.get_model(input_shape=input_shape, output_shape=output_shape)
print(model.print_summary(device="cpu"))

## load mlflow model
import mlflow.pytorch
mlflow_model_path = r"/workspace/mlflow/378794452446859122/034225c1ea9f44b598cb1b57b9d16c31/artifacts/models"
mlflow_model = mlflow.pytorch.load_model(mlflow_model_path)


predictor = Predictor(
    preprocessing_func=preprocessing, 
    postprocessing_func=postprocessing,
    model=mlflow_model)
# predictor.load_model_and_weights(model_path, device="cuda")



prediction = predictor.predict(data)
probability_map = predictor.prediction_probability_map(data)


from my_packages.neural_network.large_scans.patch_predictors import HfieldScan_Predictor
patch_shape = (30, 30)
certainty_level = 0.2
hfield_predictor = HfieldScan_Predictor(
    predictor, Hx_processed, Hy_processed, 
    patch_xbounds=patch_xbounds, patch_ybounds=patch_ybounds,
    patch_shape=patch_shape, fill_value=fill_value_for_padding, certainty_level=certainty_level
    )

hfield_predictor.plot_predictions()
plt.show()

# fig, ax = plt.subplots(3,2, figsize=(15,4.5), constrained_layout=True)
# predictor.plot(data[3], certainty_level=0.5, ax= ax[1:])
# plt.show()

print("finished")