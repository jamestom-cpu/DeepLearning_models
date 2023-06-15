import numpy as np
import matplotlib.pyplot as plt

from typing import Union, Iterable, Tuple, List
from matplotlib import patches
import matplotlib.ticker as ticker


class Cell():
    def __init__(self, center_point: Iterable[float], dimensions: Iterable[float]):
        self.center_point = np.array(list(center_point))
        self.dimensions = np.array(list(dimensions))
        self.anchor_point = self.center_point-self.dimensions/2 
        self.set_matplotlib_rectangle_artist(linewidth=1, alpha=0.7, edgecolor="red", facecolor="none")
        self.__name__ = "Cell"

    def set_matplotlib_rectangle_artist(self, **kargs):
        self.plt_patch = patches.Rectangle(self.anchor_point, *self.dimensions, **kargs)
        return self

    def __add__(self, other):
        if isinstance(other, Cell):
            return CellArray.initiate_from_list_of_cells([self, other])
            
        elif isinstance(other, CellArray): 
            new_list = [self]+list(other.cells.ravel())
            return CellArray.initiate_from_list_of_cells(new_list)
        else: 
            raise TypeError("unsupported operand type(s) for +: 'Vector' and '{}'".format(type(other).__name__))


class CellArray():
    def __init__(self, center_points: Iterable[Iterable[float]], dimensions: Iterable[Iterable[float]]):
        self.center_points = np.array(center_points)
        self.dimensions = np.array(dimensions)
        self.cell_list = self._create_cell_list()
        self.anchor_points = np.stack([cell.anchor_point for cell in self.cell_list], axis=0)
        self._shape = (len(self.cell_list))
        self.cells = np.array(self.cell_list, dtype=Cell).reshape(self._shape)
        self.__name__ = "CellArray"

    @property
    def bounds(self) -> Tuple[float, float]:
        x, y = self.dimensions
        xbound = np.sum(x, axis=0).max()
        ybound = np.sum(y, axis=1).max()
        return (xbound, ybound)

    @bounds.setter
    def bounds(self):
        return
        
    @property
    def shape(self):
        return self._shape
    
    @shape.setter
    def shape(self, shape):
        assert np.prod(np.array(list(shape))) == len(self.cell_list), f"cannot reshape {len(self.cell_list)} elements into {(shape)}"
        self._shape = shape
        self.cells = self.cells.reshape(shape)
        dimensionality = self.center_points.shape[-1]
        self.center_points = self.center_points.reshape(*shape, dimensionality)
        self.dimensions = self.dimensions.reshape(*shape, dimensionality)
        self.anchor_points = self.anchor_points.reshape(*shape, dimensionality)
        
        # put the x, y in first position
        self.center_points = np.moveaxis(self.center_points, -1, 0)
        self.dimensions = np.moveaxis(self.dimensions, -1, 0)
        self.anchor_points = np.moveaxis(self.anchor_points, -1, 0)

    def reshape(self, shape):
        self.shape  = shape
        return self
    
    def __len__(self):
        return len(self.cells)
    
    def __getitem__(self, index):
        cells = self.cells[index]
        if isinstance(cells, Iterable):
            return CellArray.initiate_from_list_of_cells(cells)
        if isinstance(cells, Cell):
            return cells

    def __setitem__(self, index, cell: Cell):
        self.cells[index] = np.array(cell)
    
    def __add__(self, other):
        if isinstance(other, Cell):
            other_cell_array = self.initiate_from_list_of_cells(other)
            return self.__add__(other_cell_array)
        elif isinstance(other, CellArray): 
            new_list = list(self.cells.ravel())+list(other.cells.ravel())
            return self.initiate_from_list_of_cells(new_list)
        else: 
            raise TypeError("unsupported operand type(s) for +: 'Vector' and '{}'".format(type(other).__name__))


    def _create_cell_list(self):
        return np.array([Cell(c, d) for c,d in zip(self.center_points, self.dimensions)])
    
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
    def initiate_from_list_of_cells(cell_list: Union[Iterable[Cell], Cell]):
        if isinstance(cell_list, Iterable):
            cell_list = np.array(list(cell_list))
        else:
            cell_list = np.array([cell_list])
        
    
        shape = cell_list.shape
        center_points = np.stack(
            [cell.center_point for cell in cell_list.flatten()], 
            axis=0
            )
        dimensions = np.stack(
            [cell.dimensions for cell in cell_list.flatten()], 
            axis=0)
            
        return CellArray(center_points, dimensions).reshape(shape)
    
    @staticmethod
    def intiate_uniform_cell_grid(xbounds: Iterable, ybounds: Iterable, cell_layout: Tuple):
        centers = get_center_points_of_cells_over_2D(xbounds, ybounds, *cell_layout).T
        dims = get_uniform_cell_dimensions(xbounds, ybounds, *cell_layout).T
        cell_array = CellArray(centers, dims)
        cell_array.shape = cell_layout
        return cell_array

    


def get_center_points_of_segments_over_1D(bounds, N):
    separation_lines = np.linspace(*bounds, N+1)
    segments_center = [(a+b)/2 for a,b in zip(separation_lines[:-1], np.roll(separation_lines, -1)[:-1])]
    return np.array(segments_center)

def get_center_points_of_cells_over_2D(xbounds, ybounds, nx, ny, return_as_grid=False):
    # take the half points in each cell
    cell_centers = np.meshgrid(
        get_center_points_of_segments_over_1D(xbounds, nx), get_center_points_of_segments_over_1D(ybounds, ny),
        indexing="ij"
        )
    if return_as_grid:
        return np.array(cell_centers)
    
    Xc, Yc = cell_centers
    return np.stack((Xc.ravel(), Yc.ravel()), axis=0)

def get_uniform_cell_anchors(xbounds, ybounds, nx, ny):
    x_lines = np.linspace(*xbounds, nx+1)
    y_lines = np.linspace(*ybounds, ny+1)

    xx, yy = np.meshgrid(x_lines[:-1], y_lines[:-1], indexing="ij")
    anchors = np.stack([xx.ravel(), yy.ravel()], axis=0)

    return anchors

def get_uniform_cell_dimensions(xbounds, ybounds, nx, ny):
    x_lines = np.linspace(*xbounds, nx+1)
    y_lines = np.linspace(*ybounds, ny+1)

    xdims = [(b-a) for a,b in zip(x_lines[:-1], np.roll(x_lines, -1)[:-1])]
    ydims = [(b-a) for a,b in zip(y_lines[:-1], np.roll(y_lines, -1)[:-1])]

    xx_dims, yy_dims = np.meshgrid(xdims, ydims, indexing="ij")
    dims = np.stack([xx_dims.ravel(), yy_dims.ravel()], axis=0)

    return dims
