from typing import Tuple, Self, Iterable
import os
import numpy as np
import pickle


import matplotlib.pyplot as plt
from dataclasses import dataclass

from .dipole_cells import DipoleCellArray
from .cells import get_center_points_of_cells_over_2D, get_uniform_cell_dimensions
from my_packages.classes.dipole_array import FlatDipoleArray


class Mask(np.ndarray):
    def __new__(cls, mask: Iterable) -> None:
        obj = np.asarray(mask).view(cls)
        obj.Ndipoles = np.sum(np.asarray(mask, dtype=int))
        obj.__name__ = f"Mask {np.asarray(mask).shape}: {obj.Ndipoles} dipoles"
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        self.Ndipoles = getattr(obj, 'Ndipoles', np.sum(np.asarray(obj, dtype=int)))
        self.__name__ = getattr(obj, '__name__', f"Mask {np.asarray(obj).shape}: {self.Ndipoles} dipoles")


    
    def __str__(self):
        return self.__name__
    
    def __repr__(self):
        return self.__name__
    

    def to_pickle(self, varname: str, directory: str="."):

        # uniform separators
        directory = directory.replace('/', '\\')

        # if directory doesn't exist create it
        if not os.path.exists(directory):
            os.makedirs(directory)

        # check if the varname ends with the file extension
        if not any(varname.endswith(ext) for ext in [".pkl", ".pickle"]):
            varname += ".pkl"
        fullpath = os.path.join(directory, varname)

        #save array to pickle file
        pickle.dump(self, open(fullpath, "wb"))
        return fullpath

    @staticmethod 
    def from_pickle(filepath: str)->Self:
        arr = pickle.load(open(filepath, "rb"))
        return Mask(arr)
    @staticmethod
    def init_from_orientations(layer: np.ndarray, orientation: str)->Self:
        assert orientation in "xyz", AssertionError(f"Orientation must be one of 'x', 'y', 'z' or a mix, but not {orientation}")

        if len(orientation)==1:
            # single orientation
            if len(layer) != 1:
                # single layer
                layer = [layer]
            assert layer[0].ndim == 2, AssertionError(f"Layer must be 2D, not {layer[0].ndim}D")
        
        if len(orientation)==2:
            # double orientation
            assert len(layer) == 2, AssertionError(f"Layer must be 2D, not {len(layer)}D")
            assert layer[0].ndim == 2, AssertionError(f"Layer must be 2D, not {layer[0].ndim}D")
            assert layer[1].ndim == 2, AssertionError(f"Layer must be 2D, not {layer[1].ndim}D")

        # final check
        assert len(layer) == len(orientation), AssertionError(f"Layer and orientation must have the same length, not {len(layer)} and {len(orientation)}")

        zero_layer = np.zeros_like(layer[0])

        zero_dict = {"x": zero_layer, "y": zero_layer, "z": zero_layer}
        mask_dict = {orient: lay for orient, lay in zip(orientation, layer)}

        complete_dict = zero_dict | mask_dict

        return np.stack([complete_dict[ors] for ors in "xyz"], axis=0).view(Mask)

        

@dataclass
class OrientedArrays():
    all: DipoleCellArray
    x: DipoleCellArray
    y: DipoleCellArray
    z: DipoleCellArray




class DipoleCellGrid():
    def __init__(self, bounds: Tuple[Tuple[float, float]], shape: Tuple, mask: np.ndarray =None):

        centers = get_center_points_of_cells_over_2D(*bounds, *shape)
        dims = get_uniform_cell_dimensions(*bounds, *shape)

        self.center_points = centers.reshape((2, *shape))
        self.dimensions = dims.reshape((2, *shape))
        self._shape = shape
        self.orientations = "xyz"

        """
        dipole cell grids automatically include all orientations.
        I keet the first dimension as the orientation dim
        """

        orientations = np.stack([np.full(shape, ors) for ors in "x,y,z".split(",")], axis=0)
        all_cpoints = np.stack([centers]*3, axis=0)
        all_dimensions = np.stack([dims]*3, axis=0)

        # create a dipole array for every orientation
        ornt_arrays = []
        for ii in range(3):
            cpoints = all_cpoints[ii].reshape(2, -1)
            dims = all_dimensions[ii].reshape(2, -1)
            ornt = orientations[ii].ravel()

            ornt_arrays.append(DipoleCellArray.init_from_individual_lists(cpoints.T, dims.T, ornt).reshape(shape))
        

        xarray, yarray, zarray = ornt_arrays
        # create the total available cells
        self.cells = np.stack(ornt_arrays, axis=0)

        # create the "background" arrays
        self.BackgroundArrays = OrientedArrays(
            all= DipoleCellArray(self.cells.ravel()).reshape((3, *shape)),
            x = xarray, y=yarray, z=zarray
        )

        
        
        # available cells
        if mask is None:
            mask = np.ones_like(self.cells, dtype="bool").view(Mask)
        self.update_mask(mask)
        

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, other):
        return

    @property
    def mask(self):
        return self._mask
    
    @mask.setter
    def mask(self, other):
        return

    def __len__(self):
        return len(self.cells.ravel())

    def update_mask(self, new_mask: Mask):
        assert new_mask.shape == self.cells.shape, AssertionError(f"""The shape of the mask must be consistent 
        with the shape of the cells: cell.shape = {self.cells.shape} vs new_mask.shape = {new_mask.shape}
        """)
        if isinstance(new_mask, np.ndarray):
            new_mask = new_mask.astype(bool).view(Mask)
        self._mask = new_mask
        self.active_cells = self.cells[self.mask] if np.any(self.mask) else np.array([])

        self.ActiveArrays = OrientedArrays(
            all = self.BackgroundArrays.all[self.mask],
            x = self.BackgroundArrays.x[self.mask[0]],
            y = self.BackgroundArrays.y[self.mask[1]],
            z = self.BackgroundArrays.z[self.mask[2]] 
        )
        return self

    def generate_random_mask(self, number_active_dipoles: int, inplace=True):
        new_mask = create_random_mask(self.shape, number_active_dipoles, dims=3).view(Mask)
        if inplace:
            self.update_mask(new_mask)
        return new_mask
    
    def plot_cellspace(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12,4))
        set_background(self.BackgroundArrays.x, ax) 
        xlim, ylim = self.BackgroundArrays.x.limits

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return ax

    def plot_setup(self, ax = None, title=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12,4))
        set_background(self.BackgroundArrays.x, ax) 
        # any orientation is equivalent to set the background as it is ignored in the function
        
        self.ActiveArrays.all.set_rectangle_artist(linewidth=1.2, linestyle="-",
        edgecolor="blue", alpha=1, facecolor="#006400").plot_cell_array(ax=ax, marker="")
        # plot the active sources
        
        zscatter = scatter_plot_active(self.ActiveArrays.z, ax=ax, alpha = 1, marker="s", color="b", s=200) #z
        xscatter = scatter_plot_active(self.ActiveArrays.x, ax=ax, alpha = 1, marker=r">", color="r", s=100) #x
        yscatter = scatter_plot_active(self.ActiveArrays.y, ax=ax, alpha = 1, marker=r"^", color="k", s=100) #y

        scatter_list = [xscatter, yscatter, zscatter]
        label_list = [f"{direction} oriented" for direction in "xyz"]
        
        legend_properties = {'weight':'bold'}

        ax.legend(scatter_list, label_list,
            markerscale=0.8, fontsize=9, prop=legend_properties,
            bbox_to_anchor=(1.01, 0.5), loc="center left", borderaxespad=0
            )
        xlim, ylim = self.BackgroundArrays.x.limits

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if title is None:
            title = "Dipole Layout"
        ax.set_title(title, fontsize=16, weight="bold")
        return ax, scatter_list
        
        

        

def create_random_mask(shape2D: Tuple, N_dipoles: int, dims: int = 1) -> np.ndarray:
    m, n = shape2D
    # Create a m x n array with all elements equal to zero
    # remember to set the dtype to "bool" so that it can be used to select the Cell Array
    arr = np.zeros((dims, m, n), dtype="bool")
    # Generate a list of p random indices (row, column)
    idx = np.random.choice(dims * m * n, N_dipoles, replace=False)
    # Convert the list of indices to row, column pairs
    dim ,row, col = np.unravel_index(idx, (dims, m, n))
    # Set the values at the specified indices to 1
    arr[dim, row, col] = 1
    return arr


def scatter_plot_active(active_cells: DipoleCellArray, ax: plt.Axes, **kargs):
    points = active_cells.center_points.reshape(2, -1)
    return ax.scatter(*points, **kargs)
    
    

def set_background(cell_ar: DipoleCellArray, ax=None):
    """This function ignores the orientation"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,4))

    cell_ar.set_rectangle_artist(
    linewidth=0.6, linestyle=":",
    edgecolor="blue", alpha=0.3, facecolor="#006400"
    ).plot_cell_array(
        ax=ax,marker="x", centerpoint_color="k", s=0.5)