from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from my_packages.neural_network.preprocessing.fieldmaps import FieldmapPreprocessor
from my_packages.classes.field_classes import Grid, Scan
from my_packages.neural_network.large_scans.patch_extractor import ScanPatchExtractor, SlidingWindowExtractor
from my_packages.neural_network.predictor.predictor  import Predictor
from .patch_predictor_base import HfieldScan_Predictor_Base


class HfieldScan_SimplePatchPredictor(HfieldScan_Predictor_Base):
    def _init_patch_extractors(self, fill_value)-> Tuple[ScanPatchExtractor, ScanPatchExtractor]:
        # initialize the patch extractors
        self.patch_extractor_Hx = ScanPatchExtractor(
            self.Hx_scan, patch_shape=self.patch_shape, 
            patch_xbounds=self.patch_xbounds, patch_ybounds=self.patch_ybounds,
            padding_fill_value=fill_value)
        self.patch_extractor_Hy = ScanPatchExtractor(
            self.Hy_scan, patch_shape=self.patch_shape, 
            patch_xbounds=self.patch_xbounds, patch_ybounds=self.patch_ybounds,
            padding_fill_value=fill_value)
        return self.patch_extractor_Hx, self.patch_extractor_Hy
    

class HfieldScan_SlidingWindow(HfieldScan_Predictor_Base):
    def __init__(
            self, predictor: Predictor, Hx_scan: Scan, Hy_scan: Scan, 
            stride: Tuple[float, float],
            patch_xbounds: Tuple[float, float] = None, patch_ybounds: Tuple[float, float] = None, 
            patch_shape: Tuple[int, int] = None, 
            fill_value: float = 0, certainty_level: float = 0.5):
        self.stride = stride
        super().__init__(predictor, 
                         Hx_scan, 
                         Hy_scan, 
                         patch_xbounds, patch_ybounds, 
                         patch_shape, fill_value, certainty_level)
        

    def _init_patch_extractors(self, fill_value)-> Tuple[SlidingWindowExtractor, SlidingWindowExtractor]:
        xstride = self.stride[0]; ystride = self.stride[1]
        self.patch_extractor_Hx = SlidingWindowExtractor(
            self.Hx_scan, patch_shape=self.patch_shape, 
            patch_xbounds=self.patch_xbounds, patch_ybounds=self.patch_ybounds,
            padding_fill_value=fill_value, stride=(xstride, ystride))
        self.patch_extractor_Hy = SlidingWindowExtractor(
            self.Hy_scan, patch_shape=self.patch_shape, 
            patch_xbounds=self.patch_xbounds, patch_ybounds=self.patch_ybounds,
            padding_fill_value=fill_value, stride=(xstride, ystride))
        return self.patch_extractor_Hx, self.patch_extractor_Hy