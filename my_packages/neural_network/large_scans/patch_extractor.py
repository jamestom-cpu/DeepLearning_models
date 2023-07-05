from typing import Tuple, Iterable
import numpy as np
import matplotlib.pyplot as plt
from my_packages.classes.field_classes import Grid, Scan

from my_packages.neural_network.preprocessing.fieldmaps import FieldmapPreprocessor
from .patch_extractor_base import BasePatchExtractor


class ScanPatchExtractor(BasePatchExtractor):    
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
    

class SlidingWindowExtractor(BasePatchExtractor):
    def __init__(
            self, scan: Scan, 
            patch_shape: Tuple[int, int] = (30,30),
            patch_xbounds: Tuple[float, float] = (-1e-2, 1e-2),
            patch_ybounds: Tuple[float, float] = (-1e-2, 1e-2),
            padding_fill_value: float = 0.0,
            stride: Tuple[float, float] = (5e-3, 5e-3)):
        super().__init__(scan, patch_shape, patch_xbounds, patch_ybounds, padding_fill_value)
        self.stride = stride
        self.xstride = stride[0]
        self.ystride = stride[1]

        assert self.xstride <= self.patch_xbounds[1] - self.patch_xbounds[0]
        assert self.ystride <= self.patch_ybounds[1] - self.patch_ybounds[0]

    def get_patches(self):
        xmin, xmax = self.scan.grid.x.min(), self.scan.grid.x.max()
        ymin, ymax = self.scan.grid.y.min(), self.scan.grid.y.max()

        # Size of the patch
        patch_size_x = self.patch_xbounds[1] - self.patch_xbounds[0]
        patch_size_y = self.patch_ybounds[1] - self.patch_ybounds[0]

        # missing area covered by the patches
        x_missing = (xmax - xmin - patch_size_x)%self.xstride
        y_missing = (ymax - ymin - patch_size_y)%self.ystride

        # Extra area covered by the patches
        x_extra = self.xstride - x_missing
        y_extra = self.ystride - y_missing

        # Update the scan boundaries to add padding if necessary
        new_scan_xbounds = (xmin - x_extra/2, xmax + x_extra/2)
        new_scan_ybounds = (ymin - y_extra/2, ymax + y_extra/2)

        # Calculate the number of steps in each direction
        nx = int(np.round((np.ptp(new_scan_xbounds)-patch_size_x) / self.xstride, 0)) + 1
        ny = int(np.round((np.ptp(new_scan_ybounds)-patch_size_y) / self.ystride, 0)) + 1

        # get padded scan
        self._original_scan = self.scan
        self.scan = self.scan.return_padded_scan(
            new_scan_xbounds, new_scan_ybounds, fill_value=self.padding_fill_value)
        
        patches = np.empty((nx, ny), dtype=Scan)
        for ix in range(nx):
            for iy in range(ny):
                # Determine the position of the patch
                x0_ii = new_scan_xbounds[0] + ix * self.xstride
                y0_ii = new_scan_ybounds[0] + iy * self.ystride

                # Determine the bounds for this patch
                xbounds = (x0_ii, x0_ii + patch_size_x)
                ybounds = (y0_ii, y0_ii + patch_size_y)

                # Get the patch
                patches[ix, iy] = self.get_patch(xbounds, ybounds)

        # restore the original scan
        self.padded_scan = self.scan
        self.scan = self._original_scan

        return patches
