import os, sys
import numpy as np
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from my_packages.model.classes.cells import Cell, CellArray
from my_packages.classes.dipole_array import FlatDipoleArray

class DipoleCell(Cell):
    def __init__(self, center_point: Iterable[float], dimensions: Iterable[float], orientation: str):
        assert orientation in ["x", "y", "z"], "orientation must be either 'x', 'y' or 'z'"
        super().__init__(center_point, dimensions)
        self.orientation = orientation
    
    def __add__(self, other):
        assert isinstance(other, DipoleCell) or isinstance(other, DipoleCellArray), TypeError(f"unsupported type: {type(other)}")
        result = super().__add__(other)
        if isinstance(other, DipoleCell):
            orientation_list = [self.orientation]+[other.orientation]
        if isinstance(other, DipoleCellArray):
            orientation_list = [self.orientation]+other.orientations
        return DipoleCellArray.init_from_CellArray(result, orientation_list)



class DipoleCellArray():    
    def __init__(
        self, 
        cells: Iterable[DipoleCell], 
        ):
        self.orientations = np.array([cell.orientation for cell in cells], dtype=str) if len(cells)>0 else np.array([])
        self.center_points = np.stack([cell.center_point for cell in cells], axis=1) if len(cells)>0 else np.array([])
        self.dimensions = np.stack([cell.dimensions for cell in cells], axis=1) if len(cells)>0 else np.array([])
        self.cells = np.array(cells, dtype=DipoleCell).ravel() if len(cells)>0 else np.array([])
        self.cell_list = list(self.cells.flatten())
        self.__name__ = "DipoleCellArray"
    
    @property
    def shape(self):
        return self.cells.shape
    
    @shape.setter
    def shape(self, new_shape):
        return
    
    @property
    def limits(self) -> Tuple[Tuple[float, float]]:
        cpointsx, cpointsy  = self.center_points.reshape(2, -1)
        dimsx, dimsy = self.dimensions.reshape(2, -1)


        min_index_x = np.argmin(cpointsx)
        max_index_x = np.argmax(cpointsx)

        minx = cpointsx[min_index_x]-dimsx[min_index_x]/2
        maxx = cpointsx[max_index_x]+dimsx[max_index_x]/2

        min_index_y = np.argmin(cpointsy)
        max_index_y = np.argmax(cpointsy)

        miny = cpointsy[min_index_y]-dimsy[min_index_y]/2
        maxy = cpointsy[max_index_y]+dimsy[max_index_y]/2

        return (minx, maxx), (miny, maxy)
    
    @property
    def bounds(self) -> Tuple[float, float]:
        x, y = self.dimensions
        xbound = np.sum(x, axis=0).max()
        ybound = np.sum(y, axis=1).max()
        return (xbound, ybound)

    @bounds.setter
    def bounds(self):
        return
    
    def __getitem__(self, index):
        cells = self.cells[index]
        if isinstance(cells, Iterable):
            return DipoleCellArray(cells)
        if isinstance(cells, DipoleCell):
            return cells

    def __setitem__(self, index, cell: DipoleCell):
        self.cells[index] = np.array(cell)
    

    def __add__(self, other):
        if isinstance(other, DipoleCell):
            other = DipoleCellArray([other])
        elif isinstance(other, DipoleCellArray):
            pass
        else:
            raise TypeError(f"We can only add with DipoleCell or DipoleCellArray. You have type {type(other)}")     
        new_list = np.concatenate([self.cells, other.cells ])        
        return DipoleCellArray(new_list)
    
    def __len__(self):
        return len(self.cells)
    
    def reshape(self, new_shape: Tuple):
        self.cells = self.cells.reshape(new_shape)
        self.orientations = self.orientations.reshape(new_shape)
        self.center_points = self.center_points.reshape(2, *new_shape)
        self.dimensions = self.dimensions.reshape(2, *new_shape)
        return self

    def generate_dipole_array(self, height: float, f: float, moments: np.ndarray = None) -> FlatDipoleArray:
        orientations=self.orientations.flatten()
        orient_dict = dict(x=[np.pi/2,0], y=[np.pi/2, np.pi/2], z=[0,0])

        num_orientations = np.stack([orient_dict[ornt] for ornt in orientations], axis=0)
        r0 = self.center_points.reshape(2, -1).T

        return FlatDipoleArray(f=f, height=height, r0=r0, orientations=num_orientations, moments=moments)
    
    # plotting funcs
    def set_rectangle_artist(self, **kargs):
        for cell in self.cell_list:
            cell.set_matplotlib_rectangle_artist(**kargs)
        return self

    def plot_cell_array(self, ax=None, centerpoint_color = "b", marker=".", space_units = "mm", **kargs):
        if ax is None:
            fig, ax = plt.subplots()
            self.ax = ax
            self.fig = fig
    
        for cell in self.cell_list:
            ax.add_patch(cell.plt_patch)
        
        if space_units == "mm":
            unit_factor = 1e-3
        else:
            unit_factor = 1 
        
        ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/unit_factor))
        ax.xaxis.set_major_formatter(ticks)
        ax.yaxis.set_major_formatter(ticks)
        
        ax.scatter(*self.center_points.reshape(self.center_points.shape[0], -1), marker=marker, color=centerpoint_color, **kargs)
        ax.set_xlabel("x[mm]")
        ax.set_ylabel("y[mm]")
        return self
    
    @staticmethod
    def init_from_CellArray(cell_arr: CellArray, orientations: Iterable[str]):
        dipole_cell_list = [DipoleCell(cell.center_point, cell.dimensions, orient) for cell, orient in zip(cell_arr.cells, orientations)]
        return DipoleCellArray(dipole_cell_list)
    
    @staticmethod
    def init_from_individual_lists(center_points: Iterable[Iterable[float]], dimensions: Iterable[Iterable[float]], orientations: Iterable[str]):
        cell_list = [DipoleCell(cpoint, dim, ornt) for cpoint, dim, ornt in zip(center_points, dimensions, orientations)]
        return DipoleCellArray(cell_list)