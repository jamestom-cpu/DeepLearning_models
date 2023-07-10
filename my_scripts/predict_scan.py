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
from singleton_python_objects.NN_models_empty.Upsampling import get_model



# load the data
db_directory = "/ext_data/simulated_fields"
# db_name = "field_testing_for_susceptibility.hdf5"
db_name = "fields_for_testing.h5"
# savename = "RSTEST_2_LINEs.hdf5"
savename = "strip_with_patch"

field_loader = FieldLoader(db_directory, db_name, savename, probe_height=6e-3)

Hx, Hy, Ez = field_loader.Hx, field_loader.Hy, field_loader.Ez




# fig, ax = plt.subplots(1,2, figsize=(10,5), constrained_layout=True)
# Hx.plot_fieldmap(ax=ax[0])
# Hy.plot_fieldmap(ax=ax[1])
# plt.show()

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

# fig, ax = plt.subplots(2,2, figsize=(10,5), constrained_layout=True)
# Hx.plot(ax=ax[0,0])
# Hy.plot(ax=ax[1,0])

# Hx_processed.plot(ax=ax[0,1])
# Hy_processed.plot(ax=ax[1,1])

# plt.show()

## Extract patches from the fieldmaps
from my_packages.neural_network.large_scans.patch_extractor import ScanPatchExtractor, SlidingWindowExtractor
        
        

fill_value_for_padding = 0.0
patch_xbounds = (-2e-2, 2e-2)
patch_ybounds = (-2e-2, 2e-2)

patch_extractor_Hx = SlidingWindowExtractor(
    Hx_processed, patch_xbounds=patch_xbounds, patch_ybounds=patch_ybounds,
    patch_shape=(30,30), padding_fill_value=fill_value_for_padding, stride=(5e-3, 5e-3)
    )
patch_extractor_Hy = SlidingWindowExtractor(
    Hy_processed, patch_xbounds=patch_xbounds, patch_ybounds=patch_ybounds,
    patch_shape=(30,30), padding_fill_value=fill_value_for_padding, stride=(5e-3, 5e-3)
    )


# patch_extractor_Hx = ScanPatchExtractor(
#     Hx_processed, patch_xbounds=patch_xbounds, patch_ybounds=patch_ybounds,
#     patch_shape=(30,30), padding_fill_value=fill_value_for_padding)
# patch_extractor_Hy = ScanPatchExtractor(
#     Hy_processed, patch_xbounds=patch_xbounds, patch_ybounds=patch_ybounds,
#     patch_shape=(30,30), padding_fill_value=fill_value_for_padding)

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

# ## plot all patches
# patch_extractor_Hx.plot_patches(Hx_patches, use_01_minmax=True)
# patch_extractor_Hy.plot_patches(Hy_patches, use_01_minmax=True)

# fig, ax = plt.subplots(figsize=(10,2), constrained_layout=True)
# Hx_patches[1,0].plot_fieldmap(ax=ax)
# plt.show()

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

model = get_model(input_shape=input_shape, output_shape=output_shape)
print(model.print_summary(device="cpu"))

## load mlflow model
import mlflow.pytorch
# mlflow_model_path = r"/workspace/mlflow/378794452446859122/034225c1ea9f44b598cb1b57b9d16c31/artifacts/models"
mlflow_model_path = "/workspace/mlflow/378794452446859122/1259b37442ad4ca6be6bcac076adddb9/artifacts/models"
mlflow_model = mlflow.pytorch.load_model(mlflow_model_path)


predictor = Predictor(
    preprocessing_func=preprocessing, 
    postprocessing_func=postprocessing,
    model=mlflow_model)
# predictor.load_model_and_weights(model_path, device="cuda")



prediction = predictor.predict(data)
probability_map = predictor.prediction_probability_map(data)


from my_packages.neural_network.large_scans.patch_predictors import HfieldScan_SimplePatchPredictor, HfieldScan_SlidingWindow
patch_shape = (30, 30)
certainty_level = 0.2
stride = (1e-3, 2e-3)
hfield_predictor = HfieldScan_SlidingWindow(
    predictor, Hx_processed, Hy_processed, stride = stride,
    patch_xbounds=patch_xbounds, patch_ybounds=patch_ybounds,
    patch_shape=patch_shape, fill_value=fill_value_for_padding, certainty_level=certainty_level
    )

#%%
xpatches, ypatches = hfield_predictor.Hx_patches, hfield_predictor.Hy_patches
xpred, ypred = hfield_predictor.x_predictions, hfield_predictor.y_predictions
xprobmap, yprobmap = hfield_predictor.x_probability_maps, hfield_predictor.y_probability_maps

prediction_scans_x = np.empty_like(xpatches, dtype=Scan)
prediction_scans_y = np.empty_like(ypatches, dtype=Scan)

# get patch prediction grid
def get_patch_grid(patch: Scan, prediction_shape: Tuple[int, int]):
    xbounds, ybounds = patch.grid.bounds()[:2]
    xshape, yshape = prediction_shape
    xstep = np.ptp(xbounds)/xshape
    ystep = np.ptp(ybounds)/yshape
    x = np.arange(xbounds[0], xbounds[1], xstep) + xstep/2
    y = np.arange(ybounds[0], ybounds[1], ystep) + ystep/2
    meshgrid = np.asarray(np.meshgrid(x, y, [0],  indexing="ij"))[..., 0]
    return Grid(meshgrid)

def get_prediction_scans(xpatches, ypatches, xpred, ypred):
    for i in range(xpatches.shape[0]):
        for j in range(xpatches.shape[1]):
            sx = xpatches[i,j]
            sy = ypatches[i,j]
            prediction_scans_x[i,j] = Scan(
                grid=get_patch_grid(sx, xpred.shape[2:]),
                scan=xpred[i,j], 
                freq=sx.f,
                axis=sx.axis, 
                component=sx.component, 
                field_type=sx.field_type, 
                )
            prediction_scans_y[i,j] = Scan(
                grid=get_patch_grid(sy, ypred.shape[2:]),
                scan=ypred[i,j], 
                freq=sy.f,
                axis=sy.axis, 
                component=sy.component, 
                field_type=sy.field_type, 
                )
    return prediction_scans_x, prediction_scans_y
        
def return_prediction_list(pred_scans: Iterable[Scan]):
    list_of_predictions = []
    for scan in pred_scans.flatten():
        print("bounds: ", scan.grid.bounds())
        print("")
        for ii in range(scan.grid.shape[1]):
            for jj in range(scan.grid.shape[2]):
                x = scan.grid.x[ii] 
                y = scan.grid.y[jj]
                print("x: ", x, "y: ", y)
                value = scan.scan[ii,jj]

                list_of_predictions.append(
                    [x, y, value])
    return np.asarray(list_of_predictions).T



# fig, ax = plt.subplots(1,2, figsize=(10,5), constrained_layout=True)
# ax[0].scatter(*list_of_predictions_x[:2], c=list_of_predictions_x[2], cmap="jet")
# ax[1].scatter(*list_of_predictions_y[:2], c=list_of_predictions_y[2], cmap="jet")
# plt.show()




from scipy.stats import binned_statistic_2d
def create_2d_histogram(x, y, v, xbounds, ybounds, output_shape):
    # Define bins
    xbins = np.linspace(xbounds[0], xbounds[1], output_shape[0] + 1) 
    ybins = np.linspace(ybounds[0], ybounds[1], output_shape[1] + 1)

    # Compute the center of the bins
    xbin_centers = (xbins[:-1] + xbins[1:]) / 2
    ybin_centers = (ybins[:-1] + ybins[1:]) / 2
    
    # Compute histogram
    ret = binned_statistic_2d(x, y, v, statistic='mean', bins=[xbins, ybins], expand_binnumbers=True)
    ret = np.nan_to_num(ret.statistic, nan=0.0)
    return ret, xbin_centers, ybin_centers

def return_combined_scan(input_scan, list_of_predictions, output_shape):
    xbounds, ybounds = input_scan.grid.bounds()[:2]
    hist, xbin_centers, ybin_centers = create_2d_histogram(*list_of_predictions, xbounds, ybounds, output_shape)
    grid = np.asarray(np.meshgrid(xbin_centers, ybin_centers, [0],  indexing="ij"))[..., 0]
    return Scan(
        grid=Grid(grid), scan=hist, freq=input_scan.f, 
        axis=input_scan.axis, component=input_scan.component, 
        field_type=input_scan.field_type)

output_shape = (31,25)
predictions = (xprobmap, yprobmap)
prediction_scans_x, prediction_scans_y = get_prediction_scans(xpatches, ypatches, *predictions)
list_of_predictions_x = return_prediction_list(prediction_scans_x)
list_of_predictions_y = return_prediction_list(prediction_scans_y)
pred_x = return_combined_scan(Hx_processed, list_of_predictions_x, output_shape)    
pred_y = return_combined_scan(Hy_processed, list_of_predictions_y, output_shape)

fig, ax = plt.subplots(1,2, figsize=(10,5), constrained_layout=True)
pred_x.plot_fieldmap(ax=ax[0])
pred_y.plot_fieldmap(ax=ax[1])
ax[0].set_title("x oriented dipoles")
ax[1].set_title("y oriented dipoles")
plt.show()

hfield_predictor.plot_predictions()
plt.show()

# fig, ax = plt.subplots(3,2, figsize=(15,4.5), constrained_layout=True)
# predictor.plot(data[3], certainty_level=0.5, ax= ax[1:])
# plt.show()

print("finished")