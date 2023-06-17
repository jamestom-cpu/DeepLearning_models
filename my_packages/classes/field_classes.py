import numpy as np
import pandas as pd
import seaborn as sns
from copy import copy
from scipy import interpolate
from matplotlib import pyplot as plt
from dataclasses import dataclass
from functools import wraps

from my_packages.classes import plot_fields
from my_packages.auxillary_math_functions import curl
from my_packages.format_checkers import check_filter_format
from .aux_classes import Grid

from typing import Tuple, Iterable, List, Union, Callable, Dict

from IPython.utils import io





class Field2D(plot_fields.Field2D_plots):
    def __init__(self, field:np.ndarray, freqs:np.ndarray, grid: Grid, axis: str, name=None): 
        self.field = field
        self.freqs = np.asarray(freqs)
        self.grid = Grid(grid)

        assert axis in ["x", "y", "z"], "axis must be x, y or z"
        self.axis = axis
        self.__name__ = name
        
    @property
    def complex_norm(self):
        return np.linalg.norm(np.real(self.field), axis=0) + 1j*np.linalg.norm(np.imag(self.field), axis=0)
    @property
    def shape(self):
        return self.field.shape
    
    @property
    def norm(self) -> np.ndarray:
        return np.linalg.norm(self.field, axis=0)
    
    @property
    def min(self) -> float:
        return self.norm.min()
    @property
    def max(self) -> float:
        return self.norm.max()
    
    @shape.setter
    def shape(self, other):
        print("cannot set shape")

    def __abs__(self)-> "Field2D":
        return Field2D(np.abs(self.field), self.freqs, self.grid, self.axis)
    


    def check_double_operation(method):
        def wrapper(self: 'Field2D', other: Union['Field3D', 'Field2D', 'PartialField', int, float, complex, np.ndarray]):

            if isinstance(self, PartialField):
                self3D = self.to_3D_field()
                method_3d = getattr(self3D, method.__name__)
                return method_3d(other)
            
            if isinstance(other, Field2D):
                same_grid = np.allclose(other.grid, self.grid)
                same_freqs = np.allclose(other.freqs, self.freqs)
                same_field_shape = np.all(other.field.shape == self.field.shape)
                same_axis = self.axis == other.axis

                assert all([same_grid,same_freqs]), f"The fields must have same grid and same freqs: \
                same grid: {same_grid}, same freqs: {same_freqs}, same field shape: {same_field_shape}, \
                same axis: {same_axis}"

                return method(self, other.field)
            
            if isinstance(other, Field3D):
                other_axis = getattr(other.grid, self.axis)
                assert len(other_axis)==1, f"The field must have only one point in the {self.axis} direction"
                self3d = self.to_3D_field()
                method3d = getattr(self3d, method.__name__)
                return method3d(other)


            elif isinstance(other, (int, float, complex, np.ndarray)):
                return method(self, other)

            else:
                raise TypeError("unsupported operand type(s) for +: 'Vector' and '{}'".format(type(other).__name__))

        return wrapper
    @check_double_operation
    def __truediv__(self, other):
        return Field2D(self.field/other, self.freqs, self.grid, self.axis)
    
    def __floordiv__(self, other):
        """create a shortcut to create a class with same properties, but different field values"""
        if isinstance(other, np.ndarray):
            assert(other.shape == self.field.shape)
            return Field2D(other, self.freqs, self.grid, self.axis)
        if isinstance(other, Field2D):
            same_grid = np.all(other.grid == self.grid)
            same_freqs = np.all(other.freqs == self.freqs)
            same_field_shape = np.all(other.field.shape==self.field.shape)
            same_axis = self.axis == other.axis

            assert all([same_grid,same_freqs]), f"The fields must have same grid and same freqs: \
            same grid: {same_grid}, same freqs: {same_freqs}, same field shape: {same_field_shape}, \
                same axis: {same_axis}"
            
            return Field2D(other.field, self.freqs, self.grid, self.axis)
        else:
            raise TypeError("Invalid argument type for field reassignment: ", type(other))        

    
    @check_double_operation
    def __mul__(self, other):
        return Field2D(self.field*other, self.freqs, self.grid, self.axis)
        
    @check_double_operation
    def __add__(self, other):
        return Field2D(self.field+other, self.freqs, self.grid, self.axis)
        
    @check_double_operation
    def __sub__(self, other):
        return Field2D(self.field-other, self.freqs, self.grid, self.axis)

    
    # make Field2D subsctiptable
    def __getitem__(self, index: int)->np.ndarray:
        return self.field[index]
        
    def __setitem__(self, index: int, field_value: np.ndarray)-> None:
        self.field[index] = np.asarray(field_value)

    
    def to_3D_field(self)-> 'Field3D':
        """convert a 2D field to a 3D field by adding a z axis"""

        grid3d = Grid(np.meshgrid(*self.grid.axes(), indexing='ij')) 
        field3d = np.expand_dims(self.field, axis={"x":1, "y":2, "z":3}[self.axis])
        return Field3D(field3d, self.freqs, grid3d)
    
    def normalized_field(self, max_min : Tuple[float, float]=None)-> 'Field2D':
        if max_min is None:
            max_min = (self.max, self.min)
        max, min = max_min
        norm_field = (self.field-min)/(max-min)
        return Field2D(norm_field, self.freqs, self.grid, self.axis)
    
    
    def get_grid_axis(self):
        X, Y, _ = self.grid
        return X[:,0], Y[0,:]
        
    def get_grid_boundaries(self):
        X, Y, _ = self.grid
        xmin, xmax = X[:,0].min(), X[:,0].max()
        ymin, ymax = Y[0,:].min(), Y[0,:].max()

        return ([xmin, xmax], [ymin, ymax])

    def resample_on_grid(self, grid)-> 'Field2D':
        # first flatten the grid to have a list of points
        flatten_grid = np.stack([grid[0].flatten(), grid[1].flatten()], axis=1)
        # evaluate the the field on each point
        field_points = self.evaluate_at_points(flatten_grid)
        #reshape the field points to the original shape
        field_shape = tuple([field_points.shape[1]]+list(grid.shape[1:])+list(self.freqs.shape))
        field_on_grid = field_points.swapaxes(0,1).reshape(field_shape)
        return Field2D(field_on_grid, self.freqs, grid, self.axis)

    def resample_on_shape(self, shape)-> 'Field2D':
        b1, b2 = [b for b in self.grid.bounds() if b[0]!=b[1]]

        N1, N2 = shape

        axis1 = np.linspace(b1[0], b1[1], N1)
        axis2 = np.linspace(b2[0], b2[1], N2)
        axis1D = getattr(self.grid, self.axis)

        if self.axis=="x":
            new_grid = np.asarray(np.meshgrid(axis1D, axis1, axis2, indexing="ij"))[:, 0, :, :]
        elif self.axis=="y":
            new_grid = np.asanyarray(np.meshgrid(axis1, axis1D, axis2, indexing="ij"))[:, :, 0, :]
        elif self.axis=="z":
            new_grid = np.asanyarray(np.meshgrid(axis1, axis2, axis1D, indexing="ij"))[:, :, :, 0]

        new_grid = Grid(new_grid)
        return self.resample_on_grid(new_grid)

    def evaluate_at_frequencies(self, freqs):
        coeff_func = interpolate.interp1d(self.freqs, self.field, axis=-1)
        new_field = coeff_func(freqs)

        
        return new_field
    
    def return_partial_field(self, components="xyz")->"PartialField2D":
        components = components.lower()
        components_indices = []
        if "x" in components:
            components_indices.append(0)
        if "y" in components:
            components_indices.append(1)
        if "z" in components:
            components_indices.append(2)
        if components_indices == []:
            raise ValueError("components must be a list of strings, a string that specifies either in 'x,y,z'")

        partial_field = self.field[components_indices, ...]
        return PartialField2D(partial_field, self.freqs, self.grid, components=components, axis=self.axis, name=self.__name__)
    
    def evaluate_at_points(self, points)-> np.ndarray:
        axes = list(filter(lambda x: len(x)!=1, self.grid.axes()))
        
        assert len(axes)==2, "The field must be 2D"

        ax1, ax2 = axes
        field_values = []
        
        for ii, _ in enumerate(self.freqs):
            fn = [interpolate.RegularGridInterpolator((ax1, ax2), self.field[comp, :, :, ii]) for comp in range(len(self.field))]
            
            
            # remember the random points are not in mm
            field_values.append(np.stack([fn_cc(points) for fn_cc in fn], axis=1))
        
        field_values = np.stack(field_values, axis=-1)
        return field_values
    
    def get_field_on_line(self, axis_line="x", axis_position=0, axis_points=None) -> 'FieldonLine':
        # cannot plot in the direction of the 2D field axis
        field_axis = self.axis
        if axis_line == field_axis:
            raise ValueError(f"axis {axis_line} is the same as field axis {field_axis}")

        if axis_points is None:
            axis_points = get_default_points(self, axis=axis_line)

        # clever interpolation looking at the field 2D axis
        coordinates = get_point_coordinates(self, axis_line, axis_points, axis_position)   
        field_values = self.evaluate_at_points(coordinates)
        # return field_values[..., f_index] if f_index is not None else field_values
        return FieldonLine(field=np.swapaxes(field_values, 0, 1), freqs=self.freqs, points = axis_points, axis_name=axis_line)

    def run_scan(self, component="z", field_type=None, f_index=0)->"Scan":
        assert field_type in ["H", "E", None], "field_type must be either 'H' or 'E'"
        if field_type is None:
            if self.__name__ == "H":
                field_type = "H"
            elif self.__name__ == "E":
                field_type="E"

        partial_field = self.return_partial_field(component)
        scan = partial_field.run_scan(field_type=field_type, f_index=f_index)
        return scan

    def plot_field_screenshot(self, f_index, ax=None, **kargs):
        fig=None
        if ax is None:
            fig, ax = plt.subplots(3,1, figsize=(12,5))

        sns.heatmap(np.real(self.field[-1, ::-1,::-1, f_index]).T, ax=ax[0], xticklabels=False, yticklabels=False, **kargs)
        ax[0].set_title("z direction")


        sns.heatmap(np.real(self.field[0, ::-1,::-1, f_index]).T, ax=ax[1], xticklabels=False, yticklabels=False, **kargs)
        ax[1].set_title("x direction")

        sns.heatmap(np.real(self.field[1, ::-1,::-1, f_index]).T, ax=ax[2], xticklabels=False, yticklabels=False, **kargs)
        ax[2].set_title("y direction")
    
    
    
    def plot_field_magnitude(self, f_index, ax=None, **kargs):
        fig=None
        if ax is None:
            fig, ax = plt.subplots(3,1, figsize=(12,5))
        

        sns.heatmap(np.abs(self.field[-1, ::-1,::-1, f_index]).T, ax=ax[0], xticklabels=False, yticklabels=False, **kargs)
        ax[0].set_title("z direction")


        sns.heatmap(np.abs(self.field[0, ::-1,::-1, f_index]).T, ax=ax[1], xticklabels=False, yticklabels=False, **kargs)
        ax[1].set_title("x direction")

        sns.heatmap(np.abs(self.field[1, ::-1,::-1, f_index]).T, ax=ax[2], xticklabels=False, yticklabels=False, **kargs)
        ax[2].set_title("y direction")
    




    


class Field3D(plot_fields.Field3D_plots):

    def __new__(cls, field: np.ndarray, freqs: np.ndarray, grid: Grid, name:str = None, *args, **kargs):
        if len(field) < 3:
            return super().__new__(PartialField)
        else:
            return super().__new__(cls)

    def __init__(self, field: np.ndarray, freqs: np.ndarray, grid: Grid, name:str = None):
        if len(field)==0:
            raise ValueError("the field is empty")
        self.field = np.array(field)
        self.freqs = freqs
        self.grid = Grid(grid)
        self.__name__ = name
        assert self.valid_grid(), "the grid is has no x or y dimensionality - maybe the indexing is wrong"

    def __getnewargs__(self):
        # return a tuple of arguments that will be passed to __new__ when unpickling
        return self.field, self.freqs, self.grid, self.__name__
    
    @property
    def shape(self):
        return self.field.shape
    
    @property
    def norm(self) -> np.ndarray:
        return np.linalg.norm(self.field, axis=0)
    
    @property
    def min(self) -> float:
        return self.norm.min()
    @property
    def max(self) -> float:
        return self.norm.max()
    
    @shape.setter
    def shape(self, other):
        print("cannot set shape")
    
    def check_double_operation(method):
        def wrapper(self: Union['Field3D', 'PartialField'], other: Union['Field3D', 'PartialField']):

            if isinstance(self, PartialField):
                if isinstance(other, Field3D) and not isinstance(other, PartialField):
                    other_partial =  other.return_partial_field(self.components)
                    return wrapper(self, other_partial)

            if isinstance(other, PartialField):
                if isinstance(self, Field3D) and not isinstance(self, PartialField):
                    self_partial =  self.return_partial_field(other.components)
                    return wrapper(self_partial, other)

                
            if isinstance(self, PartialField) and isinstance(other, PartialField):
                same_components = set(other.components) == set(self.components)
                assert same_components, f"The fields must have same components: self: {self.components} vs other: {other.components}"
                return method(self, other.field)
            
            if isinstance(other, Field3D):
                same_grid = np.allclose(other.grid, self.grid)
                same_freqs = np.allclose(other.freqs, self.freqs)
                same_field_shape = np.all(other.field.shape == self.field.shape)

                assert all([same_grid, same_freqs]), f"The fields must have same grid and same freqs: \
                same grid: {same_grid}, same freqs: {same_freqs}, same field shape: {same_field_shape}"
                return method(self, other.field)
            
            if isinstance(other, Field2D):
                other_axis = getattr(self.grid, other.axis)
                assert len(other_axis)==1, f"The field must have only one point in the {other.axis} direction"
                other3d = other.to_3D_field()
                return wrapper(self, other3d)


                # axis_height = other_axis[0]

                # self_2D = self.get_2D_field(axis=other.axis, index=axis_height)

                # # all checks taken care of by 2D sum
                # method_2d = getattr(self_2D, method.__name__)
                # version_2D = method_2d(other.field)

                # # now we need to put it back into the 3D field
                # return version_2D.to_3D_field()


            elif isinstance(other, (int, float, complex, np.ndarray)):
                return method(self, other)

            else:
                raise TypeError("unsupported operand type(s) for +: 'Vector' and '{}'".format(type(other).__name__))

        return wrapper
    
    @check_double_operation
    def __truediv__(self, other: np.ndarray):
        return Field3D(self.field / other, self.freqs, self.grid, self.__name__)
    
    @check_double_operation
    def __floordiv__(self, other: np.ndarray):
        return Field3D(other, self.freqs, self.grid, self.__name__)
        # """create a shortcut to create a class with same properties, but different field values"""
        # if isinstance(other, np.ndarray):
        #     assert(other.shape == self.field.shape)
        #     return Field3D(other, self.freqs, self.grid)
        # if isinstance(other, Field3D):
        #     same_grid = np.allclose(other.grid, self.grid)
        #     same_freqs = np.allclose(other.freqs, self.freqs)
        #     same_field_shape = np.all(other.field.shape==self.field.shape)

        #     assert all([same_grid,same_freqs]), f"The fields must have same grid and same freqs: \
        #     same grid: {same_grid}, same freqs: {same_freqs}, same field shape: {same_field_shape}"
           
            
        #     return Field3D(other.field, self.freqs, self.grid)
        # else:
        #     raise TypeError("Invalid argument type for field reassignment: ", type(other))        

    
    @check_double_operation
    def __mul__(self, other):
        return Field3D(self.field*other, self.freqs, self.grid, self.__name__)
        
    
    @check_double_operation
    def __pow__(self, other: np.ndarray)->'Field3D':
        return Field3D(self.field**other, self.freqs, self.grid, self.__name__)


    @check_double_operation
    def __add__(self, other):
        return Field3D(self.field+other, self.freqs, self.grid, self.__name__)
        
    @check_double_operation    
    def __sub__(self, other):
        return Field3D(self.field-other, self.freqs, self.grid, self.__name__)
    
    # make Field3D subscriptable
    def __getitem__(self, key: int)-> np.ndarray:
        return self.field[key]

    def __setitem__(self, key: int, value: np.ndarray)-> None:
        self.field[key] = value

    def __abs__(self):
        return Field3D(np.abs(self.field), self.freqs, self.grid, self.__name__)


    def mean(self):
        return np.mean(self.field)

    def valid_grid(self):
        xbounds, ybounds, zbounds = self.get_grid_boundaries()

        xcondition = xbounds[1] > xbounds[0]
        ycondition = ybounds[1] > ybounds[0]
        zcondition = zbounds[1] > zbounds[0]

        return any([xcondition, ycondition]) 

    def curl(self):
        curl_field = curl(self.field, *self.get_grid_axis())
        return Field3D(curl_field, self.freqs, self.grid, self.__name__)      

    def get_2D_field(self, axis="z", index=0):

        if type(index) in [float, np.float64, np.float32]:
            xaxis, yaxis, zaxis = self.get_grid_axis()
            if axis == "x":
                new_3d_grid = Grid(np.meshgrid([index], yaxis, zaxis, indexing="ij"))
                flatfield = self.resample_on_grid(new_3d_grid, overwrite=False)
                return Field2D(flatfield.field[:, 0, :, :, :], flatfield.freqs, flatfield.grid[:, 0, :, :], axis, self.__name__)
            if axis == "y":
                new_3d_grid = Grid(np.meshgrid(xaxis, [index], zaxis, indexing="ij"))
                flatfield = self.resample_on_grid(new_3d_grid, overwrite=False)
                return Field2D(flatfield.field[:, :, 0, :, :], flatfield.freqs, flatfield.grid[:, :, 0, :], axis, self.__name__)
            if axis == "z":
                new_3d_grid = Grid(np.meshgrid(xaxis, yaxis, [index], indexing="ij"))
                flatfield = self.resample_on_grid(new_3d_grid, overwrite=False)
                return Field2D(flatfield.field[:, :,:, 0, :], flatfield.freqs, flatfield.grid[:, :,:, 0], axis, self.__name__)

        if type(index) is int:
            if axis == "x":
                return Field2D(self.field[:, index, :, :, :], self.freqs, self.grid[:, index, :, :], axis, self.__name__)
            if axis == "y":
                return Field2D(self.field[:, :, index, :, :], self.freqs, self.grid[:, :, index, :], axis, self.__name__)
            if axis == "z":
                return Field2D(self.field[:, :,:, index, :], self.freqs, self.grid[:, :,:, index], axis, self.__name__)
        
        else:
            raise TypeError("Invalid index type: ", type(index))

    def evaluate_at_points(self, points):
        points = np.asarray(points)
        x_axis, y_axis, z_axis = self.grid.x, self.grid.y, self.grid.z
        field_values = []

        for ii, f in enumerate(self.freqs):
            if len(x_axis) > 1 and len(y_axis) > 1 and len(z_axis) > 1:  # 3D case
                fn = [interpolate.RegularGridInterpolator((x_axis, y_axis, z_axis), self.field[comp, :, :, :, ii]) for comp in range(len(self.field))]
            elif len(x_axis) > 1 and len(y_axis) > 1:  # 2D case, X-Y plane
                assert np.allclose(points[:, 2], z_axis[0]), "the z coordinates of the points must be equal to the value of z_axis: {}".format(z_axis[0])
                fn = [interpolate.RegularGridInterpolator((x_axis, y_axis), self.field[comp, :, :, 0, ii]) for comp in range(len(self.field))]
                points = np.asarray(points)[:, :2]
            elif len(x_axis) > 1 and len(z_axis) > 1:  # 2D case, X-Z plane
                assert np.allclose(points[:, 1], y_axis[0]), "the y coordinates of the points must be equal to the value of y_axis: {}".format(y_axis[0])
                fn = [interpolate.RegularGridInterpolator((x_axis, z_axis), self.field[comp, :, 0, :, ii]) for comp in range(len(self.field))]
                points = np.asarray(points)[:, [0, 2]]
            elif len(y_axis) > 1 and len(z_axis) > 1:  # 2D case, Y-Z plane
                assert np.allclose(points[:, 0], x_axis[0]), "the x coordinates of the points must be equal to the value of x_axis: {}".format(x_axis[0])
                fn = [interpolate.RegularGridInterpolator((y_axis, z_axis), self.field[comp, 0, :, :, ii]) for comp in range(len(self.field))]
                points = np.asarray(points)[:, 1:]
            elif len(x_axis) > 1:  # 1D case, X axis
                fn = [interpolate.interp1d(x_axis, self.field[comp, :, 0, 0, ii]) for comp in range(len(self.field))]
                points = np.asarray(points)[:, 0]
            elif len(y_axis) > 1:  # 1D case, Y axis
                fn = [interpolate.interp1d(y_axis, self.field[comp, 0, :, 0, ii]) for comp in range(len(self.field))]
                points = np.asarray(points)[:, 1]
            else:  # 1D case, Z axis
                fn = [interpolate.interp1d(z_axis, self.field[comp, 0, 0, :, ii]) for comp in range(len(self.field))]
                points = np.asarray(points)[:, 2]
            # remember the random points are not in mm
            field_values.append(np.stack([fn_cc(points).copy(order="C") for fn_cc in fn], axis=1))

        field_values = np.stack(field_values, axis=-1)

        return field_values  
    
    
    def return_partial_field(self, components="xyz") -> 'PartialField':
        components = components.lower()
        components_indices = []
        if "x" in components:
            components_indices.append(0)
        if "y" in components:
            components_indices.append(1)
        if "z" in components:
            components_indices.append(2)
        if components_indices == []:
            raise ValueError("components must be a list of strings, a string that specifies either in 'x,y,z'")

        partial_field = self.field[components_indices, ...]
        return PartialField(partial_field, self.freqs, self.grid, components=components, name = self.__name__)
        
    
    def evaluate_at_frequencies(self, freqs, overwrite=True):
        if isinstance(freqs, float):
            freqs = [freqs]
        if len(self.freqs) == 1 and len(freqs)==1:
            assert self.freqs[0] == freqs[0], "the field has only one frequency, and it is not the same as the one you are trying to evaluate: {} vs {}".format(self.freqs[0], freqs[0])
            return self
        coeff_func = interpolate.interp1d(self.freqs, self.field, axis=-1)
        new_field = coeff_func(freqs)

        if overwrite:
            self.freqs = freqs
            self.field = new_field
            return self
        
        return Field3D(new_field, freqs, self.grid, self.__name__)


    def resample_on_grid(self, grid, overwrite=True):
        # first flatten the grid to have a list of points
        flatten_grid = np.stack([grid[0].flatten(), grid[1].flatten(), grid[2].flatten()], axis=1)
        # evaluate the the field on each point
        field_points = self.evaluate_at_points(flatten_grid)
        #reshape the field points to the original shape
        field_shape = tuple([field_points.shape[1]]+list(grid.shape[1:])+list(self.freqs.shape))
        # print("field points: ", field_points.shape)
        # print("field shape: ", field_shape)
        field_on_grid = field_points.swapaxes(0,1).reshape(field_shape)
        if overwrite:
            new_self = self
        else:
            return Field3D(field_on_grid, self.freqs, grid, self.__name__)

    def get_grid_axis(self):
        X, Y, Z = self.grid
        return X[:,0,0], Y[0,:,0], Z[0,0,:]
        
    def get_grid_boundaries(self):
        return self.grid.bounds()

    def resample_at_height(self, index, axis="z"):
        if type(index) is float:      
            xaxis, yaxis, zaxis = self.get_grid_axis()
            if axis == "x":
                new_3d_grid = Grid(np.meshgrid([index], yaxis, zaxis, indexing="ij"))
                flatfield = self.resample_on_grid(new_3d_grid, overwrite=False)
                return Field3D(flatfield.field, flatfield.freqs, flatfield.grid, self.__name__)
            if axis == "y":
                new_3d_grid = Grid(np.meshgrid(xaxis, [index], zaxis, indexing="ij"))
                flatfield = self.resample_on_grid(new_3d_grid, overwrite=False)
                return Field3D(flatfield.field, flatfield.freqs, flatfield.grid, self.__name__)
            if axis == "z":
                new_3d_grid = Grid(np.meshgrid(xaxis, yaxis, [index], indexing="ij"))
                flatfield = self.resample_on_grid(new_3d_grid, overwrite=False)
                return Field3D(flatfield.field, flatfield.freqs, flatfield.grid, self.__name__)

        if type(index) is int:
            if axis == "x":
                return Field3D(self.field[:, [index], :, :, :], self.freqs, self.grid[:, [index], :, :], self.__name__)
            if axis == "y":
                return Field3D(self.field[:, :, [index], :, :], self.freqs, self.grid[:, :, [index], :], self.__name__)
            if axis == "z":
                return Field3D(self.field[:, :,:, [index], :], self.freqs, self.grid[:, :,:, [index]], self.__name__)

    def resample_on_shape(self, shape, overwrite=False):
        xN, yN, zN = shape
        xlims, ylims, zlims = self.get_grid_boundaries()

        xaxis = np.linspace(xlims[0], xlims[1], xN)
        yaxis = np.linspace(ylims[0], ylims[1], yN)
        zaxis = np.linspace(zlims[0], zlims[1], zN)

        new_grid = np.meshgrid(xaxis, yaxis, zaxis, indexing="ij")
        new_grid = Grid(new_grid)
        return self.resample_on_grid(new_grid, overwrite)




    def to_time_domain(self, t=0, f_index=0):
        # f(t) = Re{Fexp(jwt)}

        f = self.freqs[f_index]

        rotating_vector = np.exp(1j*2*np.pi*f*np.array(t))
        complex_vector = np.multiply(rotating_vector.T, np.expand_dims(self.field[...,f_index], axis=0).T).T
        new_field = np.moveaxis(np.real(complex_vector), 0, -1)
        return Field3D(new_field, t, self.grid, self.__name__)
    
    
    
    def normalized_field(self, max_min : Tuple[float, float]=None)-> 'Field3D':
        if max_min is None:
            max_min = (self.max, self.min)

        max, min = max_min
        norm_field = (self.field-min)/(max-min)
        return Field3D(norm_field, self.freqs, self.grid, self.__name__)
    
    def run_scan(self, component = "z", field_type=None, axis="z", index=0, f_index=0)->"Scan":
        assert component in "xyz", "component must be one of 'x', 'y', 'z'"
        assert axis in "xyz", "axis must be one of 'x', 'y', 'z'"
        assert field_type in ["E", "H", None], "field_type must be one of 'E', 'H', None"

        return self.get_2D_field(axis, index).run_scan(component, field_type, f_index)
        
        
        
    
    

class PartialField(Field3D):
    def __init__(
        self, field: np.ndarray, 
        freqs: np.ndarray, 
        grid: Grid, 
        components: str = None,
        name: str = None
        ):
        self.field = field
        self.freqs = freqs
        self.grid = grid
        if components is None:
            if len(self.field) == 1:
                self.components = "z"
            if len(self.field) == 2:
                self.components = "xy"
        else:
            self.components = components
        self.__name__ = name
        
    def get_2D_field(self, axis="z", index=0)-> "PartialField2D":
        field2D =  super().get_2D_field(axis, index)
        return PartialField2D(field2D.field, field2D.freqs, field2D.grid, axis=axis, components = self.components, name= self.__name__)

class PartialField2D(Field2D):
    def __init__(
        self, field: np.ndarray, 
        freqs: np.ndarray, 
        grid: Grid, 
        axis: str,
        components: str = None,
        name: str = None
        ):
        self.field = field
        self.freqs = np.asarray(freqs)
        self.grid = grid
        if components is None:
            if len(self.field) == 1:
                self.components = "z"
            if len(self.field) == 2:
                self.components = "xy"
        else:
            self.components = components
        assert axis in ["x", "y", "z"], "axis must be x, y or z"
        self.axis = axis
        self.__name__ = name

    def __abs__(self)-> "PartialField2D":
        return PartialField2D(np.abs(self.field), self.freqs, self.grid, self.axis, self.components, self.__name__)


    def resample_on_grid(self, grid: np.array) -> "PartialField2D":
        grid = Grid(grid)
        resampled_field = super().resample_on_grid(grid).field
        return PartialField2D(resampled_field, self.freqs, grid, self.axis, self.components, self.__name__)
    
    def resample_on_shape(self, shape)-> "PartialField2D":
        field2d =  super().resample_on_shape(shape)
        return PartialField2D(field2d.field, self.freqs, field2d.grid, self.axis, self.components, self.__name__)
    
    def contour_plot_magnitude(
            self, field_component="n", f_index=0, scale="linear", 
            units="A/m", ax=None, title=None, keep_linear_scale_on_colormap=False, 
            vmin=None, vmax=None, *args, **kwargs):
        assert field_component in list(self.components)+["n"], "field_component must be one of"+",".join(list(self.components)+["n"])
        
        zero_field = np.zeros_like(self.field[0])
        my_field = {comp: field_comp for comp, field_comp in zip(self.components, self.field)}

        dummy_2D_field = np.array([my_field.get(comp, zero_field) for comp in "xyz"])
        dummy_field_2D = Field2D(dummy_2D_field, self.freqs, self.grid, self.axis)

        return dummy_field_2D.contour_plot_magnitude(field_component, f_index, scale, units, ax, title, keep_linear_scale_on_colormap, vmin, vmax, *args, **kwargs)

    def contour_plot_phase(self, field_component="n", f_index=0, units="A/m", ax=None, title=None,  keep_linear_scale_on_colormap=False, vmin=None, vmax=None):
        assert field_component in list(self.components)+["n"], "field_component must be one of"+",".join(list(self.components)+["n"])
        
        zero_field = np.zeros_like(self.field[0])
        my_field = {comp: field_comp for comp, field_comp in zip(self.components, self.field)}

        dummy_2D_field = np.array([my_field.get(comp, zero_field) for comp in "xyz"])
        dummy_field_2D = Field2D(dummy_2D_field, self.freqs, self.grid, self.axis)

        return dummy_field_2D.contour_plot_phase(field_component, f_index, units, ax, title, keep_linear_scale_on_colormap, vmin, vmax)

    def run_scan(self, component=None, f_index=0, field_type=None)-> "Scan":
        assert component is None or isinstance(component, str), "component must be a string"
        if component is not None: 
            assert component in self.components, "component must be one of "+",".join(self.components)

        if component is None:
            component = self.components[0]
        
        scanned_field = np.abs(self).field[self.components.index(component), ..., f_index].squeeze()

        
        if field_type is None:
            if self.__name__ == "H":
                field_type = "H"
            else:
                field_type = "E"
        return Scan(scanned_field, self.grid, self.freqs[f_index], self.axis, component, field_type)
    
    def update_with_scan(self, scan: np.ndarray, component:str = None, f_index=0, keep_angle=False):
        if isinstance(component, str):
            assert component in self.components, "component must be one of "+",".join(self.components)
        if component is None:
            component = self.components[0]
        
        assert scan.shape == self.field[self.components.index(component), ..., f_index].shape, "scan must have the same shape as the field"
        new_field = copy(self.field)
        new_field[self.components.index(component), ..., f_index] = scan
        if keep_angle:
            new_field = np.abs(new_field)*np.exp(1j*np.angle(self.field))
        return PartialField2D(new_field, self.freqs, self.grid, self.axis, self.components, self.__name__)

    @staticmethod
    def init_from_single_scan(scan: np.ndarray, grid: Grid, axis: str, freq: float, component: str):
        assert axis in ["x", "y", "z"], "axis must be x, y or z"
        assert component in ["x", "y", "z"], "component must be x, y or z"

        field = np.expand_dims(scan, axis=[0, -1])
        return PartialField2D(field, freq, grid, axis, component)


class FieldonLine():
    def __init__(
            self,
            field: np.ndarray,
            points: np.ndarray,
            freqs: np.ndarray,
            axis_name: str | None = None,
            field_name: str | None = None,
    ):
        self.field = np.ascontiguousarray(field)
        self.points = np.ascontiguousarray(points)
        self.freqs = np.ascontiguousarray(freqs)
        self.axis_name = axis_name
        self.__name__ = field_name



    @property
    def field_norm(self):
        return np.linalg.norm(self.field, axis=0)
    @property
    def max(self):
        return np.max(self.field_norm)
    @property
    def min(self): 
        return np.min(self.field_norm)   
    def __abs__(self):
        return abs(self.field)
    
    def resample(self, N: int, kind="quadratic")-> 'FieldonLine':
        new_points = np.linspace(self.points.min(), self.points.max(), N)
        return self.resample_on_points(new_points, kind=kind)
    
    # function to resample on a new set of points
    def resample_on_points(self, new_points: list | np.ndarray, kind="quadratic")-> 'FieldonLine':
        new_points = np.ascontiguousarray(new_points)
        assert (len(new_points.shape)==1), "new_points must be a 1D array" 

        new_field = np.zeros((self.field.shape[0], new_points.shape[0], len(self.freqs)), dtype=np.complex128)
        for i in range(self.field.shape[0]):
            for f in range(len(self.freqs)):
                new_field[i, :, f] = interp1d(self.points, self.field[i, :, f], kind=kind)(new_points)
        return FieldonLine(new_field, new_points, self.freqs, self.axis_name, self.__name__)
        
    
    def plot_field(self, component="n", f_index=0, ax=None, units="H/m", title=None, show_max_values=True, **kwargs):
        if ax is None:
            _, ax = plt.subplots(figsize=(12,4), constrained_layout=True)
        if component in ("n", "norm", "normal"):
            my_field_component = self.field_norm[..., f_index]
        elif component == "x":
            my_field_component = self.field[0, ..., f_index]
        elif component == "y": 
            my_field_component = self.field[1, ..., f_index]
        elif component == "z":
            my_field_component = self.field[2, ..., f_index]
        else:
            raise ValueError(f"component {component} not recognized")  
        
        my_field_component = np.abs(my_field_component)
        
        xlabel = f"{self.axis_name} [mm]" if self.axis_name is not None else "axis points [mm]"

        default_kwargs = dict(linewidth=2, label=f"{self.freqs[f_index]/1e6:.2f} MHz")
        default_kwargs.update(kwargs)


        ax.plot(self.points*1e3, my_field_component, **default_kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"[{units}]")
        title = f"{title} - {component} Component" if title is not None else f"{component} Component"
        ax.set_title(title, fontsize=16)
        ax.grid(True)
        ax.set_xlim(self.points.min()*1e3, self.points.max()*1e3)

        ax.legend()
        if show_max_values:
            # add the max values to the y ticks
            y_ticks = ax.get_yticks().tolist()
            average_step = np.mean(np.diff(y_ticks))
            y_ticks.append(my_field_component.max())
            y_ticks = np.sort(y_ticks)

            # index of the my_field_component.max() in the y_ticks list
            ii = np.where(y_ticks == my_field_component.max())[0][0]
            
            # if the max value is too close to the previous value, remove the previous value
            if ii>0 and np.abs(y_ticks[ii-1] - y_ticks[ii]) < average_step/2:
                # remove the previous value
                y_ticks = np.delete(y_ticks, ii-1)
            
            # if the max value is too close to the next value, remove the next value
            if ii<len(y_ticks)-1 and np.abs(y_ticks[ii+1] - y_ticks[ii]) < average_step/2:
                # remove the next value
                y_ticks = np.delete(y_ticks, ii+1)

            ax.set_yticks(y_ticks)

            ii = np.argmax(my_field_component)
            # add a horizontal line at the max value from zero to the max value in the x direction
            ax.hlines(
                my_field_component.max(), xmin=self.points[0]*1e3, xmax=self.points[ii]*1e3, 
                color="k", linestyle="-.", linewidth=0.75)
            
            # props = dict(boxstyle='round4', facecolor='white', edgecolor="black", alpha=1)
            # k =0.05
            # ax.text(
            #     self.points[0]*1e3+(self.points[-1]*1e3 - self.points[0]*1e3)*k, my_field_component.max(), 
            #     f"{my_field_component.max():.2f}", fontsize=10, color="k", 
            #     verticalalignment='top', horizontalalignment="center", bbox=props)
        return ax




##########################
# auxiliary functions
##########################

def get_default_points(field2D: Field2D, axis="x") -> np.ndarray:
    # 2D grid have x y axes only, we need to translate the axis to the 2D grid
    try:
        points = getattr(field2D.grid, axis)
    except AttributeError:
        raise AttributeError(f"axis {axis} not recognized")
    
    return points

def get_point_coordinates(field2D: Field2D, axis_line:str, axis_points: np.ndarray, axis_position: float) -> np.ndarray:
    # write 3 coordinates for the points
    possible_combinations = {"x": ["y", "z"], "y": ["x", "z"], "z": ["x", "y"]}
    
    possible_axes = possible_combinations[field2D.axis]

    if axis_line not in possible_axes:
        raise ValueError(f"axis {axis_line} not recognized")
    
    coordinates = {}

    coordinates[axis_line] = axis_points
    coordinates[field2D.axis] = np.full_like(axis_points, getattr(field2D.grid, field2D.axis)[0])
    axis = next(filter(lambda x: x != axis_line, possible_axes))
    coordinates[axis] = np.full_like(axis_points, axis_position)
    
    coordinate_points = np.stack([coordinates[possible_axes[0]], coordinates[possible_axes[1]]], axis=1)
    return coordinate_points

# 1d interpolation
from scipy.interpolate import interp1d

def resample_complex_array(z: np.array, x: np.array, new_x:np.array, kind="quadratic"):
    # Create a function for interpolating the real and imaginary parts separately
    f_real = interp1d(x, z.real, kind=kind)
    f_imag = interp1d(x, z.imag, kind=kind)

    # Evaluate the function at the new values of x to get the real and imaginary parts of the resampled array
    new_real = f_real(new_x)
    new_imag = f_imag(new_x)

    # Combine the real and imaginary parts to get the resampled complex array
    new_z = new_real + 1j*new_imag

    return new_z


    

    


############################################

class Scan(PartialField2D, np.ndarray):
    """
    This class represents a 2D scan of a field. It may contain the phase information, but in general it is not used.
    I try to avoid it being used because in the long run it will not be available from measurement. 
    
    This class should simplify denoising, erosion and other operations on the field magnitude.

    This is one of the classes that should be eventually used in the dipole array reconstruction. Multiple scan classes 
    if multiple scans are needed for:
    - different components
    - different frequencies
    - different heights -> note that the Grid parameter contains information on the height of the scan

    This class is instantiated as a view of the numpy array, so it is not a copy of the data. This is important to
    avoid memory issues. The class extends the numpy array, so all the numpy array methods are available.

    Parameters
    ----------
    scan : np.ndarray
        2D array containing the field magnitude
    grid : Grid
        Grid object containing the information on the grid
    freq : float
        frequency of the scan
    axis : str, optional
        axis of the scan, by default "z"
    component : str, optional
        component of the field, by default "z"
    field_type : str, optional
        type of the field, by default "E"

    """
    def __new__(cls, scan: np.ndarray, *args, **kargs) -> None:
        assert scan.ndim == 2, "field must be 2D"
        obj = np.ascontiguousarray(scan).view(cls)
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return

        self.grid = getattr(obj, 'grid', None)
        self.field_type = getattr(obj, 'field_type', None)
        self.axis = getattr(obj, 'axis', None)

        # self.components is not used in scan. Rather self.component is used. However, self.component was defined a property
        # that points to self.components that comes from the PartialField2D class. This is done to avoid code duplication.
        #
        self.components = getattr(obj, 'components', None)
        self.field = getattr(obj, 'field', None)
        # self.scan = getattr(obj, 'scan', None)
        self.f = getattr(obj, 'f', None)
        self.freqs = getattr(obj, 'freqs', None)
    
    def __init__(self, scan: np.ndarray, grid: Grid, freq: float, axis: str = "z", component: str = "z",  field_type="E"):
        assert field_type in ["E", "H"], "field_type must be either 'E' or 'H'"
        # check the number of dimensions of the field
        assert scan.ndim == 2, "field must be 2D"
        # assert np.ndims(field) == 2, "field must be 2D"
        assert component in ["x", "y", "z"], "component must be either 'x', 'y' or 'z'"
        assert axis in ["x", "y", "z"], "axis must be x, y or z"
        # 
        new_field = np.expand_dims(scan, axis=[0,-1])

        self.f = freq
        self.field_type = field_type

        super().__init__(new_field, [freq], grid, axis=axis, components=component)

    # v is an alias for scan. In the future, v should be used. I leave scan for backward compatibility
    @property
    def v(self):
        return np.abs(self.view(np.ndarray))
    
    @v.setter
    def v(self, value):
        self.view(np.ndarray)[:] = value

    

    @property
    def scan(self):
        return np.abs(self.view(np.ndarray))
    
    @scan.setter
    def scan(self, value):
        self.view(np.ndarray)[:] = value

    
    
    @property
    def phase(self):
        return np.angle(self.view(np.ndarray)) if self.dtype is complex else np.zeros_like(self.view(np.ndarray))

    @property
    def norm(self):
        return np.linalg.norm(self.view(np.ndarray), axis=0)        

    @property
    def component(self):
        return self.components

    @property
    def shape(self):
        return self.scan.shape
    
    # set how the class should look when printed
    def __repr__(self):
        return f"Scan with shape {self.shape} and frequency {self.f}"
    
    def __str__(self):
        return self.__repr__()
    
    def __getitem__(self, key) -> "Scan":
        new_scan = self.scan[key]

        if new_scan.ndim == 1 or new_scan.ndim == 0 or isinstance(new_scan, (float, complex, int)):
            return new_scan
        
        if not hasattr(key, "__len__"):
            new_grid = self.grid[:, key]
        elif len(key) == 2:
            if isinstance(key[0], slice) and isinstance(key[1], slice):
                new_grid = self.grid[key[0], key[1]]
            else:
                indices = (slice(None),) + key
                new_grid = self.grid[indices]
        else:
            raise ValueError("key is too large: must be 1 or 2. Length of key is {}".format(len(key)))
        
        return Scan(new_scan, new_grid, self.f, axis=self.axis, component=self.component, field_type=self.field_type)
    def __setitem__(self, key, value):
        self.scan[key] = value

    def check_double_operation(method):
        @wraps(method)
        def wrapper(self: 'Scan', other: Union['Scan', int, float, complex, np.ndarray]):
            
            if isinstance(other, Scan):
                same_grid = np.allclose(other.grid, self.grid)
                same_freqs = np.allclose(other.f, self.f)
                same_field_shape = np.all(other.shape == self.shape)
                same_axis = self.axis == other.axis
                same_component = self.component == other.component
                same_field_type = self.field_type == other.field_type

                assert all([same_grid,same_freqs]), f"The fields must have same grid and same freqs: \
                same grid: {same_grid}, same freqs: {same_freqs}, same field shape: {same_field_shape}, \
                same axis: {same_axis}, same component: {same_component}, same field type: {same_field_type}"

                return method(self, other.scan)
            

            elif isinstance(other, (int, float, complex, np.ndarray)):
                return method(self, other)

            else:
                raise TypeError("unsupported operand type(s) for +: 'Vector' and '{}'".format(type(other).__name__))

        return wrapper
    
    @check_double_operation
    def __add__(self, other)->"Scan":
        """phase is lost"""
        new_scan = self.scan + other
        return Scan(new_scan, self.grid, self.f, axis=self.axis, component=self.component, field_type=self.field_type)
    
    @check_double_operation
    def __sub__(self, other)->"Scan":
        """phase is lost"""
        new_scan = self.scan - other
        return Scan(new_scan, self.grid, self.f, axis=self.axis, component=self.component, field_type=self.field_type)
    @check_double_operation
    def __mul__(self, other)->"Scan":
        """phase is lost"""
        new_scan = self.scan * other
        return Scan(new_scan, self.grid, self.f, axis=self.axis, component=self.component, field_type=self.field_type)
    @check_double_operation
    def __truediv__(self, other)->"Scan":
        """phase is lost"""
        new_scan = self.scan / other
        return Scan(new_scan, self.grid, self.f, axis=self.axis, component=self.component, field_type=self.field_type)
    
    def __neg__(self)->"Scan":
        new_scan = -self.scan
        return Scan(new_scan, self.grid, self.f, axis=self.axis, component=self.component, field_type=self.field_type)

    def normalize(self)-> "Scan":
        """normalize the scan"""
        new_scan = self.scan / np.max(self.scan) if self.scan.dtype != bool else self.scan
        return Scan(new_scan, self.grid, self.f, axis=self.axis, component=self.component, field_type=self.field_type)

    def normalize_01(self)-> "Scan":
        """normalize the scan"""
        new_scan = (self.scan - np.min(self.scan)) / (np.max(self.scan) - np.min(self.scan)) if self.scan.dtype != bool else self.scan
        return Scan(new_scan, self.grid, self.f, axis=self.axis, component=self.component, field_type=self.field_type)    

    def update_with_scan(self, new_scan:np.ndarray)->"Scan":
        """update the scan of the scan object"""
        assert new_scan.shape == self.scan.shape, "new scan must have same shape as the old one"
        return Scan(new_scan, self.grid, self.f, axis=self.axis, component=self.component, field_type=self.field_type)
    
    def sample_point(self, x:Tuple[float, float])-> np.ndarray:
        return self.evaluate_at_points([x]).squeeze()

    def apply_filter(self, filter: Callable, *args, include_scan_normalization_step = True, **kwargs)->"Scan":
        """Apply a filter to the scan

        Parameters
        ----------
        filter : filter to apply to the scan, must be a function that takes a scan and a footprint as input as first 
        two arguments and returns a filtered scan (np.ndarray) as output

        Returns
        -------
        Scan
            filtered scan
        """

        if self.scan.dtype == bool:
            include_scan_normalization_step = False

        if include_scan_normalization_step:
            my_scan = self.normalize_01().scan
            vmin, vmax = np.min(self.scan), np.max(self.scan)
        else:
            my_scan = self.scan

        
        new_scan = filter(my_scan, *args, **kwargs)

        

        if isinstance(new_scan, tuple):
            for res in new_scan: 
                if isinstance(res, np.ndarray) and res.shape==self.scan.shape:
                    new_scan = res
                    break
            assert isinstance(new_scan, np.ndarray), "The filter must return a scan of the same shape as the input scan"
                    
        # return to normal scale
        if include_scan_normalization_step:
            new_scan = vmin + (vmax-vmin)*new_scan

        return Scan(new_scan, self.grid, self.f, axis=self.axis, component=self.component, field_type=self.field_type)

    def apply_pipeline(self, pipeline: List[Tuple[Callable, Union[Dict, List, float]]], *args, include_normalization_step = True, return_raw=False, verbose=False, **kwargs)->"Scan":
        """Apply a pipeline to the scan

        Parameters
        ----------
        pipeline : pipeline to apply to the scan, the pipeline is a list of functions plus any kwargs in tuples. 
        If an element of the pipeline is a function, it should be called without arguments or kwargs. If it is a tuple,
        the first element of the tuple must be a function that takes a scan as input and returns a
        filtered scan (np.ndarray) as output. The second element of the tuple must be a dictionary of kwargs to pass 
        to the function.

        record : bool, optional
            if True, print statements to the console as running through the pipeline, by default False. It 
            is used to debug the pipeline

        Returns
        -------
        Scan
            filtered scan
        """
        new_scan = self.scan.copy()
        pipeline = check_filter_format(pipeline)
        # for element in pipeline:
        #     if isinstance(element, tuple):
        #         if len(element) == 2:
        #             function = element[0]
        #             arguments = element[1]
        #             if isinstance(arguments, dict):
        #                 f_kwargs = element[1]
        #             elif isinstance(arguments, (list, tuple)):
        #                 f_args = arguments
        #                 f_kwargs = {}
        #             else:
        #                 f_args = (arguments,)
        #                 f_kwargs = {}
        #         # allow for args to be passed in the tuple directly
        #         elif len(element) > 2:
        #             function = element[0]
        #             f_args = element[1:]
        #             f_kwargs = {}
        #     else:
        #         function = element
        #         f_kwargs = {}
        #         f_args = ()
        #     new_kwargs = f_kwargs | kwargs
        #     new_args = args + f_args

        # convert tuple to ordered dict
        def to_dict(tup):
            assert isinstance(tup, (tuple, list)), "must be a tuple or list"
            return dict(zip(range(len(tup)), tup))

        for function, f_args, f_kwargs in pipeline:
            # update the kwargs with the input kwargs and the args with the input args
            new_kwargs = f_kwargs | kwargs
            # flatten into single tuple
            new_args = tuple((to_dict(f_args) | to_dict(args)).values())

            if verbose:
                print("running function {} with kwargs {}".format(function.__name__, new_kwargs))
            # normalize the new scan between 0 and 1
            if self.scan.dtype == bool:
                include_normalization_step=False
            if include_normalization_step:
                vmin, vmax = np.min(new_scan), np.max(new_scan)
                new_scan = (new_scan - vmin) / (vmax-vmin)

            # run each function
            new_scan = function(new_scan, *new_args, **new_kwargs)

            # safaguard for returning a single value
            if isinstance(new_scan, float):
                return_raw = True
            
            # return raw option
            if return_raw:
                if include_normalization_step:
                    new_scan = vmin + (vmax-vmin)*new_scan
                return new_scan
            if isinstance(new_scan, tuple):
                for res in new_scan: 
                    if isinstance(res, np.ndarray) and res.shape==self.scan.shape:
                        new_scan = res
                        break
                assert isinstance(new_scan, np.ndarray), "The filter must return a scan of the same shape as the input scan"
            # return to normal scale
            if include_normalization_step:
                new_scan = vmin + (vmax-vmin)*new_scan
        
        return Scan(new_scan, self.grid, self.f, axis=self.axis, component=self.component, field_type=self.field_type)

    def find_threshold(self, threshold_func: Callable, *args, scan_normalization_step = True, return_normalized_threshold=False, **kwargs)->float:
        """Find the threshold for a given function

        Parameters
        ----------
        threshold_func : function that takes a scan and returns a threshold value

        Returns
        -------
        float
            threshold value
        """
        
        my_scan = self.normalize_01().scan if scan_normalization_step else self.scan
        threshold = threshold_func(my_scan, *args, **kwargs)
        if return_normalized_threshold:
            return threshold
        

        # renormalize threshold
        return np.min(self.scan) + (np.max(self.scan) - np.min(self.scan))*threshold
        

    
    def binarize(self, threshold: Union[Callable, float, np.ndarray], *args, scan_normalization_step = True, **kwargs)->"Scan":
        if isinstance(threshold, Callable):
            threshold = self.find_threshold(threshold, *args, scan_normalization_step = scan_normalization_step, **kwargs)
        
        if isinstance(threshold, np.ndarray):
            if threshold.squeeze().shape == ():
                threshold = float(threshold.squeeze())
            else:
                assert threshold.shape == self.scan.shape, "threshold must be a float or an array of the same shape as the scan"
        else:
            threshold = float(threshold)
        

        # select high values as positive
        binary_scan = self.scan >= threshold
        return Scan(binary_scan, self.grid, self.f, axis=self.axis, component=self.component, field_type=self.field_type)

    
    def plot(self, ax=None, scale="linear", title=None, keep_linear_scale_on_colormap=False, vmin=None, vmax=None, plot_binary=False, verbose=False, **kwargs):
        """simplified plot method for scan that continues from contour_plot_magnitude method of Field class"""

        def mprint(*args):
            if verbose:
                print(*args)
            else:
                pass

        mprint("starting plot")
        if self.scan.dtype == bool or self.scan.dtype == int or plot_binary: 
            mprint("plotting binary scan")
            return plot_fields.plot_2d_field_pcolor(
                self.scan, ax, self.grid, title=title, include_colorbar=False, **kwargs
            )

        if title is None:
            title = f"{self.field_type} field"

        mprint("plotting magnitude")
        return self.contour_plot_magnitude(
            field_component=self.component,
            f_index = 0,
            scale=scale,
            units = "V/m" if self.field_type == "E" else "A/m",
            ax=ax,
            title=title,
            keep_linear_scale_on_colormap=keep_linear_scale_on_colormap,
            vmin=vmin,
            vmax=vmax,
            **kwargs
        )


    def scatter_plot(self, ax=None, **kwargs):
        """plot the positions of the pixels that are True in the scan"""
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True, dpi=100)
            
        xvals = self.grid.x[np.where(self.data)[0]]*1e3
        yvals = self.grid.y[np.where(self.data)[1]]*1e3

        if self.field_type == "E" and self.component=="z":
            default_marker = "o"
        elif self.field_type=="H" and self.component=="y":
            default_marker = "^"
        elif self.field_type=="H" and self.component=="x":
            default_marker = ">"
        else:
            default_marker = "x"

        scatter_kwargs = {
            "marker": default_marker,
            "color": "k",
            "s": 20
        }

        scatter_kwargs.update(kwargs)

        ax.scatter(xvals, yvals, **scatter_kwargs)

    
    def resample_on_grid(self, grid: np.array) -> "Scan":
        pfield_2D = super().resample_on_grid(grid)
        return pfield_2D.run_scan(self.component, 0, self.field_type)
    
    def resample_on_shape(self, shape: Tuple[int, int]) -> "Scan":
        pfield_2D = super().resample_on_shape(shape)
        return pfield_2D.run_scan(self.component, 0, self.field_type)
    
