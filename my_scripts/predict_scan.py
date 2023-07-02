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





# load scanned data
class FieldLoader():
    def __init__(self, db_directory, db_filename, db_group_name, probe_height=6e-3):
        self.db = SimpleNamespace(directory=db_directory, filename=db_filename, savename=db_group_name)
        self.probe_height = probe_height
        self.run_scans_on_database_field()
    
    def run_scans_on_database_field(self):
         # hdf5 database properties
        fullpath = os.path.join(self.db.directory, self.db.filename)

        # load the database properties to mhandler
        self.m_handler = Measurement_Handler.from_h5_file(fullpath, self.db.savename)

        # create the target fields: Ez, Hx, Hy only magnitudes on a plane
        self.Ez = self.m_handler.E.run_scan("z", field_type="E", index = self.probe_height)
        self.Hx = self.m_handler.H.run_scan("x", field_type="H", index = self.probe_height)
        self.Hy = self.m_handler.H.run_scan("y", field_type="H", index = self.probe_height)

        assert self.Ez.f == self.Hx.f == self.Hy.f, "All fields must have the same frequency"
        assert np.allclose(self.Ez.grid, self.Hx.grid) and np.allclose(self.Ez.grid, self.Hy.grid), "All fields must have the same grid"
        self.f = [self.Ez.f] # assuming all fields have the same frequency
        self.scan_grid = np.expand_dims(self.Ez.grid, axis=-1)  # the scan grid is 2D, we need to add the third dimension


# load the data
db_directory = "/ext_data/simulated_fields"
db_name = "field_testing_for_susceptibility.hdf5"
savename = "RSTEST_2_LINEs.hdf5"

field_loader = FieldLoader(db_directory, db_name, savename, probe_height=6e-3)

Hx, Hy, Ez = field_loader.Hx, field_loader.Hy, field_loader.Ez




# fig, ax = plt.subplots(1,2, figsize=(10,5), constrained_layout=True)
# Hx.plot_fieldmap(ax=ax[0])
# Hy.plot_fieldmap(ax=ax[1])
# plt.show()

# preprocess the data to obtain a smooth fieldmap
import scipy.ndimage as ndi

class FieldmapPreprocessor:
    def __init__(self, fieldmap, sigma=1.0):
        self.fieldmap = fieldmap
        self.sigma = sigma
        self.min = np.min(self.fieldmap)
        self.max = np.max(self.fieldmap)

    def remove_nan(self):
        nan_elements = np.isnan(self.fieldmap)
        if np.any(nan_elements):
            self.fieldmap[nan_elements] = np.nanmean(self.fieldmap)

    def gaussian_blur(self):
        self.fieldmap = ndi.gaussian_filter(self.fieldmap, sigma=self.sigma)

    def normalize(self):
        self.fieldmap = (self.fieldmap - np.min(self.fieldmap)) / (np.max(self.fieldmap) - np.min(self.fieldmap))

    def undo_normalization(self):
        self.fieldmap = self.fieldmap * (self.max - self.min) + self.min 

    def preprocess(self):
        self.remove_nan()
        self.gaussian_blur()
        # self.normalize()
        return self.fieldmap

Hx_processed = FieldmapPreprocessor(Hx.scan).preprocess()
Hy_processed = FieldmapPreprocessor(Hy.scan).preprocess()

Hx_processed = Hx.update_with_scan(Hx_processed)
Hy_processed = Hy.update_with_scan(Hy_processed)    

# fig, ax = plt.subplots(2,2, figsize=(10,5), constrained_layout=True)
# Hx.plot_fieldmap(ax=ax[0,0])
# Hy.plot_fieldmap(ax=ax[1,0])

# Hx_processed.plot_fieldmap(ax=ax[0,1])
# Hy_processed.plot_fieldmap(ax=ax[1,1])

# fig, ax = plt.subplots(2,2, figsize=(10,5), constrained_layout=True)
# Hx.plot(ax=ax[0,0])
# Hy.plot(ax=ax[1,0])

# Hx_processed.plot(ax=ax[0,1])
# Hy_processed.plot(ax=ax[1,1])

# plt.show()

## Extract patches from the fieldmaps
class ScanPatchExtractor:
    def __init__(
            self, scan: "Scan", 
            patch_shape: Tuple[int, int] = (30,30),
            patch_xbounds: Tuple[float, float] = (-1e-2, 1e-2),
            patch_ybounds: Tuple[float, float] = (-1e-2, 1e-2),
            padding_fill_value: float = 0.0
            ):
        self.scan = scan
        self.patch_shape = patch_shape  
        self.patch_xbounds = patch_xbounds
        self.patch_ybounds = patch_ybounds
        self.padding_fill_value = padding_fill_value

    def get_indices(self, bounds):
        x_indices = np.where((self.scan.grid.x >= bounds[0][0]) & (self.scan.grid.x <= bounds[0][1]))[0]
        y_indices = np.where((self.scan.grid.y >= bounds[1][0]) & (self.scan.grid.y <= bounds[1][1]))[0]
        return slice(x_indices.min(), x_indices.max()+1), slice(y_indices.min(), y_indices.max()+1)

    def _get_raw_patch_and_grid(self, xbounds, ybounds):
        xbounds_idx, ybounds_idx = self.get_indices((xbounds, ybounds))
        return self.scan.scan[xbounds_idx, ybounds_idx], self.scan.grid.v[:, xbounds_idx, ybounds_idx]

    def get_simple_patch(self, xbounds, ybounds):
        raw_patch, raw_grid = self._get_raw_patch_and_grid(xbounds, ybounds)
        grid = Grid(raw_grid)
        new_scan = Scan(raw_patch, grid, self.scan.f, self.scan.axis, self.scan.component, self.scan.field_type)
        return new_scan
    
    def return_raw_shape_in_patch(self, xbounds, ybounds):
        raw_patch, _ = self._get_raw_patch_and_grid(xbounds, ybounds)
        return raw_patch.shape
    
    def get_patch(
            self, 
            xbounds: Tuple[float, float]=None, 
            ybounds: Tuple[float, float]=None, 
            patch_shape: Tuple[int, int]=None) -> "Scan":

        # set default patch shape
        if patch_shape is None:
            patch_shape = self.patch_shape
        if xbounds is None:
            xbounds = self.patch_xbounds
        if ybounds is None:
            ybounds = self.patch_ybounds

        patch_grid = Grid(np.meshgrid(
                np.linspace(xbounds[0], xbounds[1], patch_shape[0]), 
                np.linspace(ybounds[0], ybounds[1], patch_shape[1]), 
                self.scan.grid.z,
                indexing='ij'))
        
        
        patch = self.scan.resample_on_grid(patch_grid)
        return patch
    
    def normalize(self, vmax=None, vmin=None):
        if vmax is None:
            vmax = self.scan.scan.max()
        if vmin is None:
            vmin = self.scan.scan.min()

        self.norm_vmax = vmax
        self.norm_vmin = vmin
        self.scan = (self.scan - vmin) / (vmax - vmin)

    def undo_normalization(self):
        assert hasattr(self, "norm_vmax"), "You must normalize the scan first"
        self.scan = self.scan * (self.norm_vmax - self.norm_vmin) + self.norm_vmin
    
    
    def get_patches(self, xstride=None, ystride=None):
        if xstride is None:
            xstride = self.patch_xbounds[1] - self.patch_xbounds[0]
        if ystride is None:
            ystride = self.patch_ybounds[1] - self.patch_ybounds[0]

        xmin, xmax = self.scan.grid.x.min(), self.scan.grid.x.max()
        ymin, ymax = self.scan.grid.y.min(), self.scan.grid.y.max()

        # Extra area covered by the patches
        x_extra = xstride - (xmax - xmin)%xstride
        y_extra = ystride - (ymax - ymin)%ystride

        # Update the scan boundaries to add padding if necessary
        new_scan_xbounds = (xmin - x_extra/2, xmax + x_extra/2)
        new_scan_ybounds = (ymin - y_extra/2, ymax + y_extra/2)

        # Calculate the number of steps in each direction
        nx = int(np.round(np.ptp(new_scan_xbounds) / xstride, 0))
        ny = int(np.round(np.ptp(new_scan_ybounds) / ystride, 0))

        
        # get padded scan
        self._original_scan = self.scan
        self.scan = self.scan.return_padded_scan(
            new_scan_xbounds, new_scan_ybounds, fill_value=self.padding_fill_value)

        patches = np.empty((nx, ny), dtype=Scan)
        for ix in range(nx):
            for iy in range(ny):
                # Determine the bounds for this patch
                xbounds = (new_scan_xbounds[0] + ix * xstride, new_scan_xbounds[0] + (ix+1) * xstride)
                ybounds = (new_scan_ybounds[0] + iy * ystride, new_scan_ybounds[0] + (iy+1) * ystride)

                # Get the patch
                patches[ix, iy] = self.get_patch(xbounds, ybounds)
        
        # restore the original scan
        self.padded_scan = self.scan
        self.scan = self._original_scan
 
        return patches
    
    def plot_patches(self, patches: Iterable[Scan], use_01_minmax: bool = True):

        if not use_01_minmax:
            vmax = self.scan.scan.max()
            vmin = self.scan.scan.min()
        else:
            vmax = 1
            vmin = 0

        nx, ny = patches.shape
        fig, axes = plt.subplots(ny, nx, figsize=(nx*3, ny), constrained_layout=True)

        # Ensure axes is a 2D array even if nx or ny are 1
        if nx == 1:
            axes = axes[np.newaxis, :]
        if ny == 1:
            axes = axes[:, np.newaxis]

        for ii in range(nx):
            for jj in range(ny):
                patches[ii, jj].plot_fieldmap(ax=axes[ny-jj-1, ii], build_colorbar=False, vmin=vmin, vmax=vmax)
                # remove the axis
                axes[ny-jj-1, ii].axis('off')
                # remove the ticks
                axes[ny-jj-1, ii].set_xticks([])
                axes[ny-jj-1, ii].set_yticks([])
                # remove the title
                axes[ny-jj-1, ii].set_title("")
                # remove the labels
                axes[ny-jj-1, ii].set_xlabel("")
        fig.suptitle(f"Patches - vmax: {vmax:.2f}, vmin: {vmin:.2f}")

        return fig, axes
        
        

fill_value_for_padding = 0.0
patch_extractor_Hx = ScanPatchExtractor(Hx_processed, patch_shape=(30,30), padding_fill_value=fill_value_for_padding)
patch_extractor_Hy = ScanPatchExtractor(Hy_processed, patch_shape=(30,30), padding_fill_value=fill_value_for_padding)

# normalize 
vmax = np.max([Hx_processed.scan.max(), Hy_processed.scan.max()])
vmin = fill_value_for_padding

# normalize the batch extractors
patch_extractor_Hx.normalize(vmax=vmax, vmin=vmin)
patch_extractor_Hy.normalize(vmax=vmax, vmin=vmin)

Hx_patch_center_raw = patch_extractor_Hx.get_simple_patch(
    xbounds=[-1e-2, 1e-2], ybounds=[-1e-2, 1e-2]
    )
Hx_patch_center = patch_extractor_Hx.get_patch(
    xbounds=[-1e-2, 1e-2], ybounds=[-1e-2, 1e-2]
    )

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
# Hx_patches[3,5].plot_fieldmap(ax=ax)
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

model = Quasi_ResNet.get_model(input_shape=input_shape, output_shape=output_shape)
print(model.print_summary(device="cpu"))

## load mlflow model
import mlflow.pytorch
mlflow_model_path = r"/workspace/mlflow/829057909622999832/700803c8139d40aea15b0b6e809d0cda/artifacts/models"
mlflow_model = mlflow.pytorch.load_model(mlflow_model_path)


predictor = Predictor(
    preprocessing_func=preprocessing, 
    postprocessing_func=postprocessing,
    model=mlflow_model)
# predictor.load_model_and_weights(model_path, device="cuda")



prediction = predictor.predict(data)
probability_map = predictor.prediction_probability_map(data)


class HfieldScan_Predictor():
    def __init__(
            self, 
            predictor: Predictor,
            Hx_scan: Scan,
            Hy_scan: Scan,
            patch_shape: Tuple[int, int] = None,
            fill_value: float = 0.0
            ):
        self.predictor = predictor
        self.Hx_scan = Hx_scan
        self.Hy_scan = Hy_scan

        # initialize the patch shape
        if patch_shape is None:
            assert hasattr(self.predictor.model, "input_shape"), "You must provide a patch shape"
            patch_shape = self.predictor.model.input_shape[-2:]
        self.patch_shape = patch_shape

        # initialize the patch extractors
        self.patch_extractor_Hx = ScanPatchExtractor(
            self.Hx_scan, patch_shape=self.patch_shape, padding_fill_value=fill_value)
        self.patch_extractor_Hy = ScanPatchExtractor(
            self.Hy_scan, patch_shape=self.patch_shape, padding_fill_value=fill_value)
        
        # normalize the scan data
        self._normalization(fill_value)

        # get the patches
        self.Hx_patches = patch_extractor_Hx.get_patches()
        self.Hy_patches = patch_extractor_Hy.get_patches()

        # get the data to be fed to the neural network
        self.data = self._get_data(self.Hx_patches, self.Hy_patches)

        # run the prediction
        _prediction, _probability_map = self._run_prediction()
        patch_position_shape = self.Hx_patches.shape[:2]

        # reshape the prediction and probability map
        predictions = _prediction.reshape(patch_position_shape + _prediction.shape[1:])
        self.predictions = np.moveaxis(predictions, 2, 0)
        probability_maps = _probability_map.reshape(patch_position_shape + _probability_map.shape[1:])
        self.probability_maps = np.moveaxis(probability_maps, 2, 0)

        self.x_predictions, self.y_predictions = self.predictions
        self.x_probability_maps, self.y_probability_maps = self.probability_maps

    def _run_prediction(self, certainty_level: float = 0.5):     
        prediction = self.predictor.predict(self.data, certainty_level=certainty_level)
        probability_map = self.predictor.prediction_probability_map(self.data)
        return prediction, probability_map
    

    def _normalization(self, fill_value)->None:  
        # define normalization constants
        self.vmax = np.max([self.Hx_scan.scan.max(), self.Hy_scan.scan.max()])
        self.vmin = fill_value if fill_value == 0 else np.min([self.Hx_scan.scan.min(), self.Hy_scan.scan.min()])

        # normalize the batch extractors
        self.patch_extractor_Hx.normalize(vmax=self.vmax, vmin=self.vmin)
        self.patch_extractor_Hy.normalize(vmax=self.vmax, vmin=self.vmin)

    @staticmethod
    def _get_data(Hx_patches, Hy_patches):
        # create the data
        Hx_patches = np.array([patch.scan for patch in Hx_patches.flatten()], dtype=np.float32)
        Hy_patches = np.array([patch.scan for patch in Hy_patches.flatten()], dtype=np.float32)

        data = np.stack((Hx_patches, Hy_patches), axis=1)
        return data
    
    @staticmethod
    def scatter_plot_single_layer_prediction(input: Scan, prediction, ax, marker="o", color="k", edgecolor="w"):
        pred_x, pred_y = np.asarray(np.where(prediction == 1))+0.5

        xbounds, ybounds, _ = input.grid.bounds()

        prediction_norm_x = pred_x / prediction.shape[0]
        prediction_norm_y = pred_y / prediction.shape[1]

        # rescale the coordinates
        marker_x = (xbounds[1] - xbounds[0]) * prediction_norm_x + xbounds[0]
        marker_y = (ybounds[1] - ybounds[0]) * prediction_norm_y + ybounds[0] 

        if prediction.sum() != 0:
            print("some predictions: ", prediction.sum())

        q = ax.scatter(marker_x, marker_y, marker=marker, color=color, edgecolors=edgecolor, s=75)
        return q 
    
    def plot_predictions(self):
        self.plot_predictions_component(component="x", marker=">", color="k", edgecolor="w")
        self.plot_predictions_component(component="y", marker="^", color="r", edgecolor="k")
    
    def plot_predictions_component(self, component="x", marker="o", color="k", edgecolor="w"):

        if component == "x":
            predictions = self.x_predictions
            patches = self.Hx_patches
        
        elif component == "y":
            predictions = self.y_predictions
            patches = self.Hy_patches

        vmax = 1
        vmin = 0

        nx, ny = patches.shape
        fig, axes = plt.subplots(ny, nx, figsize=(nx*1.5, ny*0.5), constrained_layout=True)

        # Ensure axes is a 2D array even if nx or ny are 1
        if nx == 1:
            axes = axes[np.newaxis, :]
        if ny == 1:
            axes = axes[:, np.newaxis]

        for ii in range(nx):
            for jj in range(ny):
                patches[ii, jj].plot_fieldmap(ax=axes[ny-jj-1, ii], build_colorbar=False, vmin=vmin, vmax=vmax)
                self.scatter_plot_single_layer_prediction(
                    patches[ii, jj], predictions[ii, jj], axes[ny-jj-1, ii],
                    marker=marker, color=color, edgecolor=edgecolor
                    )
                # remove the axis
                axes[ny-jj-1, ii].axis('off')
                # remove the ticks
                axes[ny-jj-1, ii].set_xticks([])
                axes[ny-jj-1, ii].set_yticks([])
                # remove the title
                axes[ny-jj-1, ii].set_title("")
                # remove the labels
                axes[ny-jj-1, ii].set_xlabel("")
        fig.suptitle(f"Patches - vmax: {vmax:.2f}, vmin: {vmin:.2f}")
        return fig, axes   
        
    
    def plot_patches(self, component="x", use_01_minmax: bool = True):
        if component == "x":
            patches = self.Hx_patches
            patch_extractor = self.patch_extractor_Hx
        elif component == "y":
            patches = self.Hy_patches
            patch_extractor = self.patch_extractor_Hy
        else:
            raise ValueError("component must be x or y")
        return patch_extractor.plot_patches(patches, use_01_minmax=use_01_minmax)


patch_shape = (30, 30)
hfield_predictor = HfieldScan_Predictor(
    predictor, Hx_processed, Hy_processed, 
    patch_shape=patch_shape, fill_value=fill_value_for_padding
    )

hfield_predictor.plot_predictions()
plt.show()

fig, ax = plt.subplots(3,2, figsize=(15,4.5), constrained_layout=True)
predictor.plot(data[3], certainty_level=0.5, ax= ax[1:])
plt.show()

print("finished")