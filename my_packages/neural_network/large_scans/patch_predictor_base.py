from typing import Tuple
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from my_packages.neural_network.preprocessing.fieldmaps import FieldmapPreprocessor
from my_packages.classes.field_classes import Grid, Scan
from my_packages.neural_network.predictor.predictor  import Predictor
from .patch_extractor_base import BasePatchExtractor


class HfieldScan_Predictor_Base(ABC):
    def __init__(
            self, 
            predictor: Predictor,
            Hx_scan: Scan,
            Hy_scan: Scan,
            patch_xbounds: Tuple[float, float] = None,
            patch_ybounds: Tuple[float, float] = None,
            patch_shape: Tuple[int, int] = None,
            fill_value: float = 0.0,
            certainty_level: float = 0.5
            ):
        self.patch_xbounds = patch_xbounds
        self.patch_ybounds = patch_ybounds
        self.predictor = predictor
        self.Hx_scan = Hx_scan
        self.Hy_scan = Hy_scan
        self.certainty_level = certainty_level

        # initialize the patch shape
        if patch_shape is None:
            assert hasattr(self.predictor.model, "input_shape"), "You must provide a patch shape"
            patch_shape = self.predictor.model.input_shape[-2:]
        self.patch_shape = patch_shape

        self.patch_extractor_Hx, self.patch_extractor_Hy = self._init_patch_extractors(fill_value)
        
        # normalize the scan data
        self._normalization(fill_value)

        # get the patches
        self.Hx_patches = self.patch_extractor_Hx.get_patches()
        self.Hy_patches = self.patch_extractor_Hy.get_patches()

        # get the data to be fed to the neural network
        self.data = self._get_data(self.Hx_patches, self.Hy_patches)

        # run the prediction
        _prediction, _probability_map = self._run_prediction(self.certainty_level)
        patch_position_shape = self.Hx_patches.shape[:2]

        # reshape the prediction and probability map
        predictions = _prediction.reshape(patch_position_shape + _prediction.shape[1:])
        self.predictions = np.moveaxis(predictions, 2, 0)
        probability_maps = _probability_map.reshape(patch_position_shape + _probability_map.shape[1:])
        self.probability_maps = np.moveaxis(probability_maps, 2, 0)

        self.x_predictions, self.y_predictions = self.predictions
        self.x_probability_maps, self.y_probability_maps = self.probability_maps


    @abstractmethod
    def _init_patch_extractors(self, fill_value) -> Tuple[BasePatchExtractor, BasePatchExtractor]:
        pass



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