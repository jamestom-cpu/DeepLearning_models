import ipywidgets as widgets
from IPython.display import display
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
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
import mlflow.pytorch
import ipywidgets as widgets
from IPython.display import display
from scipy.stats import binned_statistic_2d


plt.switch_backend('TkAgg')
PROJECT_CWD = r"/workspace/"
sys.path.append(PROJECT_CWD)

os.chdir(PROJECT_CWD)

# neuratl network imports
from my_packages.neural_network.data_generators.mixed_array_generator import MixedArrayGenerator
from my_packages.neural_network.large_scans.patch_extractor import ScanPatchExtractor, SlidingWindowExtractor
from my_packages.neural_network.data_generators.iterator import DataIterator
from my_packages.neural_network.model.model_trainers.dipole_position_trainer import Trainer
from my_packages.neural_network.model.model_base import Model_Base
from my_packages.neural_network.predictor.predictor  import Predictor
from my_packages.neural_network.aux_funcs.evaluation_funcs import f1_score_np
from my_packages.neural_network.large_scans.patch_predictors import HfieldScan_SimplePatchPredictor, HfieldScan_SlidingWindow


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
import mlflow.pytorch


print("cuda available: ", torch.cuda.is_available())

## inspect the data

# import the data generator 
from NN_model_architectures.PredictDipolePosition.ResNet import get_model



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
###

class UtilClass():
    def __init__(
            self, Hx_processed: Scan, Hy_processed: Scan,
            fill_value_for_padding = 0.0,
            patch_xbounds = (-2e-2, 2e-2),
            patch_ybounds = (-2e-2, 2e-2),
            patch_shape = (30,30)
            ):
        self.Hx_processed = Hx_processed
        self.Hy_processed = Hy_processed
        self.fill_value_for_padding = fill_value_for_padding
        self.patch_xbounds = patch_xbounds
        self.patch_ybounds = patch_ybounds
        self.patch_shape = patch_shape
        self.import_model()
      

    def initialize_the_patch_extractors(self, stride):
        patch_extractor_Hx = SlidingWindowExtractor(
            Hx_processed, patch_xbounds=self.patch_xbounds, patch_ybounds=self.patch_ybounds,
            patch_shape=self.patch_shape, padding_fill_value=self.ill_value_for_padding, stride=(5e-3, 5e-3)
            )
        patch_extractor_Hy = SlidingWindowExtractor(
            Hy_processed, patch_xbounds=self.patch_xbounds, patch_ybounds=self.patch_ybounds,
            patch_shape=self.patch_shape, padding_fill_value=self.fill_value_for_padding, stride=(5e-3, 5e-3)
            )
        return patch_extractor_Hx, patch_extractor_Hy
    
    def import_model(self, model_path="/workspace/mlflow/378794452446859122/ffe6bb4b8c3845ef937a32ccd390640f/artifacts/models"):  
        self.mlflow_model = mlflow.pytorch.load_model(model_path)
        return self.mlflow_model
    
    @staticmethod 
    def _preprocessing(x):
        # get the H fields
        return x
    @staticmethod
    def _postprocessing(x):
        return 1/(1+np.exp(-x))
    
    def init_simple_predictor(self):
        predictor = Predictor(
        preprocessing_func=self._preprocessing, 
        postprocessing_func=self._postprocessing,
        model=self.mlflow_model)
        return predictor

    def init_large_predictor(self, stride=5e-3, certainty_level=0.2):
        predictor = self.init_simple_predictor()
        hfield_predictor = HfieldScan_SlidingWindow(
            predictor, Hx_processed, Hy_processed, stride = stride,
            patch_xbounds=self.patch_xbounds, patch_ybounds=self.patch_ybounds,
            patch_shape=self.patch_shape, fill_value=self.fill_value_for_padding, certainty_level=certainty_level
            )
        return hfield_predictor
    
    def predict(self, output_shape, stride = (1e-3, 2e-3), certainty_level=0.2):
        hfield_predictor = self.init_large_predictor(stride=stride, certainty_level=certainty_level)

        xpatches, ypatches = hfield_predictor.Hx_patches, hfield_predictor.Hy_patches
        xprobmap, yprobmap = hfield_predictor.x_probability_maps, hfield_predictor.y_probability_maps

        predictions = (xprobmap, yprobmap)
        prediction_scans_x, prediction_scans_y = self.get_prediction_scans(xpatches, ypatches, *predictions)
        
        list_of_predictions_x = self.return_prediction_list(prediction_scans_x)
        predictionX = self.return_combined_scan(self.Hx_processed, list_of_predictions_x, output_shape)
        
        list_of_predictions_y = self.return_prediction_list(prediction_scans_y)
        predictionY = self.return_combined_scan(self.Hy_processed, list_of_predictions_y, output_shape)

        return predictionX, predictionY

    


    @staticmethod
    def return_combined_scan(input_scan, list_of_predictions, output_shape):
        xbounds, ybounds = input_scan.grid.bounds()[:2]
        hist, xbin_centers, ybin_centers = UtilClass.create_2d_histogram(*list_of_predictions, xbounds, ybounds, output_shape)
        grid = np.asarray(np.meshgrid(xbin_centers, ybin_centers, [0],  indexing="ij"))[..., 0]
        return Scan(
            grid=Grid(grid), scan=hist, freq=input_scan.f, 
            axis=input_scan.axis, component=input_scan.component, 
            field_type=input_scan.field_type)
    
    @staticmethod
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
    
    @staticmethod
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
    
    @staticmethod
    def get_patch_grid(patch: Scan, prediction_shape: Tuple[int, int]):
        xbounds, ybounds = patch.grid.bounds()[:2]
        xshape, yshape = prediction_shape
        xstep = np.ptp(xbounds)/xshape
        ystep = np.ptp(ybounds)/yshape
        x = np.arange(xbounds[0], xbounds[1], xstep) + xstep/2
        y = np.arange(ybounds[0], ybounds[1], ystep) + ystep/2
        meshgrid = np.asarray(np.meshgrid(x, y, [0],  indexing="ij"))[..., 0]
        return Grid(meshgrid)


    @staticmethod
    def get_prediction_scans(xpatches, ypatches, xpred, ypred):
        prediction_scans_x = np.empty_like(xpatches, dtype=Scan)
        prediction_scans_y = np.empty_like(ypatches, dtype=Scan)

        for i in range(xpatches.shape[0]):
            for j in range(xpatches.shape[1]):
                sx = xpatches[i,j]
                sy = ypatches[i,j]
                prediction_scans_x[i,j] = Scan(
                    grid=UtilClass.get_patch_grid(sx, xpred.shape[2:]),
                    scan=xpred[i,j], 
                    freq=sx.f,
                    axis=sx.axis, 
                    component=sx.component, 
                    field_type=sx.field_type, 
                    )
                prediction_scans_y[i,j] = Scan(
                    grid=UtilClass.get_patch_grid(sy, ypred.shape[2:]),
                    scan=ypred[i,j], 
                    freq=sy.f,
                    axis=sy.axis, 
                    component=sy.component, 
                    field_type=sy.field_type, 
                    )
        return prediction_scans_x, prediction_scans_y
    
    @staticmethod
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

# uclass = UtilClass(Hx_processed, Hy_processed)
# pred_x, pred_y = uclass.predict(output_shape=(30,30), stride = (1e-3, 2e-3), certainty_level=0.2)

# fig, ax = plt.subplots(1,2, figsize=(10,5), constrained_layout=True)
# pred_x.plot_fieldmap(ax=ax[0])
# pred_y.plot_fieldmap(ax=ax[1])
# ax[0].set_title("x oriented dipoles")
# ax[1].set_title("y oriented dipoles")
# plt.show()




# Define the UI elements
strideX_slider = widgets.FloatSlider(
    value=1e-3,
    min=1e-4,
    max=1e-2,
    step=1e-4,
    description='Stride X:',
    readout_format='.4f',
)

strideY_slider = widgets.FloatSlider(
    value=2e-3,
    min=1e-4,
    max=1e-2,
    step=1e-4,
    description='Stride Y:',
    readout_format='.4f',
)

output_shape_X_slider = widgets.IntSlider(
    value=30,
    min=3,
    max=100,
    step=1,
    description='Output Shape X:',
)

output_shape_Y_slider = widgets.IntSlider(
    value=30,
    min=3,
    max=100,
    step=1,
    description='Output Shape Y:',
)

certainty_level_slider = widgets.FloatSlider(
    value=0.2,
    min=0.0,
    max=1.0,
    step=0.01,
    description='Certainty Level:',
)

calculate_button = widgets.Button(
    description='Calculate'
)

# Define the event handlers
def on_calculate_button_clicked(b):
    stride = strideX_slider.value, strideY_slider.value
    output_shape = output_shape_X_slider.value, output_shape_Y_slider.value
    certainty_level = certainty_level_slider.value
    
    uclass = UtilClass(Hx_processed, Hy_processed)
    pred_x, pred_y = uclass.predict(output_shape=output_shape, stride = stride, certainty_level=certainty_level)

    print("Calculations are done.")

    fig, ax = plt.subplots(1,2, figsize=(10,5), constrained_layout=True)
    pred_x.plot_fieldmap(ax=ax[0])
    pred_y.plot_fieldmap(ax=ax[1])
    ax[0].set_title("x oriented dipoles")
    ax[1].set_title("y oriented dipoles")
    plt.show()

# Connect the handlers and the UI elements
calculate_button.on_click(on_calculate_button_clicked)

# Define the UI layout
ui = widgets.VBox([
    strideX_slider,
    strideY_slider,
    output_shape_X_slider,
    output_shape_Y_slider,
    certainty_level_slider,
    calculate_button
])

# Display the UI
display(ui)
plt.show()
