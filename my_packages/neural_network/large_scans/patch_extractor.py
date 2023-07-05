from typing import Tuple, Iterable
import numpy as np
import matplotlib.pyplot as plt
from my_packages.neural_network.preprocessing.fieldmaps import FieldmapPreprocessor
from my_packages.classes.field_classes import Grid, Scan


class ScanPatchExtractor:
    def __init__(
            self, scan: Scan, 
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