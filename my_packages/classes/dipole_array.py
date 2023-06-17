import numpy as np
import pandas as pd
import math as m
import cmath 
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from collections.abc import Iterable
from typing import Union, Tuple

import copy

from my_packages.classes import dipole_source, signals
from my_packages.model.classes.cells import CellArray
from my_packages.classes.aux_classes import Grid
from my_packages.auxillary_funcs_moments import return_moments_on_grid, return_moments_on_expanded_grid

class DipoleSourceArray_Plotting():
    def plot_dipole_positions(self, grid2d, ax = None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(24,4))
        
        ax.scatter(grid2d, self.r0)



class DipoleSourceArray():
    def __init__(
        self, 
        f: Iterable,
        r0:Iterable,
        orientations: Union[Iterable, str],
        moments=None,
        type = None
    ):
        self.f = np.array(f)


        # each dipole has different values
        self.r0 = np.array(r0)
        self.N_dipoles = len(self.r0)
        self._orientations = None
        self.orientations = orientations
        if moments is None:
            moments = np.ones((self.N_dipoles, len(f)))*1e-20
        self._dipole_moments = np.array(moments)
        self._dipoles = self.get_list_of_dipoles()
        if type is not None:
            self.set_dipole_type(type)
    
    # alias for dipole moments
    @property
    def M(self):
        return self._dipole_moments
    
    @property
    def magnetic_array(self):
        magnetic_dipole_list = filter(lambda x: x.type == "Magnetic", self.dipoles)
        return DipoleSourceArray.init_dipole_array_from_dipole_list(self.f, list(magnetic_dipole_list))

    @property
    def electric_array(self):
        electric_dipole_list = filter(lambda x: x.type == "Electric", self.dipoles)
        return DipoleSourceArray.init_dipole_array_from_dipole_list(self.f, list(electric_dipole_list))


    @property
    def orientations(self):
        return self._orientations
    
    @orientations.setter
    def orientations(self, ors):
        my_dict = dict(x=[np.pi/2,0], y=[np.pi/2, np.pi/2], z=[0,0])
        if not isinstance(ors, np.ndarray) and ors in ["x", "y", "z"]: 
            ors = fix_alligned_dipole_orientation(self.N_dipoles, *my_dict[ors])
        self._orientations = np.array(ors) 

    
    @property
    def dipole_moments(self):
        return self._dipole_moments
    
    @dipole_moments.setter
    def dipole_moments(self, M):
        self._dipole_moments=np.array(M)
        self.get_list_of_dipoles()
    
    @property
    def dipoles(self):
        return self._dipoles

    @dipoles.setter
    def dipoles(self, dip:list):
        self._dipoles = dip
        self.r0 = np.array([dipole.r0 for dipole in dip]) 
        self.orientations = np.array([[dipole.theta, dipole.phi] for dipole in dip])
        self._dipole_moments = np.array([dipole.dipole_moment.signal for dipole in dip])

    def __add__(self, other):
        
        if isinstance(other, Iterable):
            if len(other) == 0:
                return self
            

        elif isinstance(other, DipoleSourceArray):
           return DipoleSourceArray.init_dipole_array_from_dipole_list(self.f, self.dipoles+other.dipoles)
        
            
        elif isinstance(other, dipole_source.Dipole_Source):
            assert other.dipole_moment.f == self.f, "the frequencies must be the same"
            return DipoleSourceArray.init_dipole_array_from_dipole_list(self.f, self.dipoles+[other])
        
        elif isinstance(other, (float, np.array)):
            new_M = self.M + other
            return self//new_M # shortcut to create a new class with same properties, but different field values and maintain the same type
            
        else: 
            raise TypeError("unsupported operand type(s) for +: 'DipoleArray' and '{}'".format(type(other).__name__))
        
    # write the same for multiplication
    def __mul__(self, other):
        if isinstance(other, Iterable):
            if len(other) == 0:
                return self

        elif isinstance(other, DipoleSourceArray):
           assert other.f == self.f, "the frequencies must be the same"
           new_M = self.M * other.M
           return self//new_M
        
            
        elif isinstance(other, dipole_source.Dipole_Source):
            assert other.dipole_moment.f == self.f, "the frequencies must be the same"
            new_M = self.M * other.dipole_moment.signal 
            return self//new_M
               
        elif isinstance(other, (int, float, np.array)):
            new_M = self.M * other
            return self//new_M
        else:
            raise TypeError("unsupported operand type(s) for *: 'DipoleArray' and '{}'".format(type(other).__name__))
    
    def __truediv__(self, other):
        if isinstance(other, Iterable):
            if len(other) == 0:
                return self

        elif isinstance(other, DipoleSourceArray):
           assert other.f == self.f, "the frequencies must be the same"
           new_M = self.M / other.M
           return self//new_M
        
            
        elif isinstance(other, dipole_source.Dipole_Source):
            assert other.dipole_moment.f == self.f, "the frequencies must be the same"
            new_M = self.M / other.dipole_moment.signal 
            return self//new_M
               
        elif isinstance(other, (int, float, np.array)):
            new_M = self.M / other
            return self//new_M
        else:
            raise TypeError("unsupported operand type(s) for /: 'DipoleArray' and '{}'".format(type(other).__name__))
    
    
    def __floordiv__(self, other):
        """create a shortcut to create a class with same properties, but different field values"""
        if isinstance(other, (np.ndarray, list)):

            other = np.array(other)

            if np.ndim(other) == 1:
                other = np.expand_dims(other, axis=1)
        

            N, F = other.shape

            assert N==self.N_dipoles and F==len(self.f), f"""the moments assigned should have the same 
            number of freqs/number of dipoles: \n
            number of freqs {F} vs {len(self.f)}\n
            number of dipoles {N} vs {self.N_dipoles}
            """
            dipole_types = [d.type for d in self.dipoles]
            return DipoleSourceArray(self.f, self.r0, self.orientations, other).set_individual_dipole_type(dipole_types)

        if isinstance(other, Iterable):         
            return DipoleSourceArray.init_dipole_array_from_dipole_list(self.f, other)
        else:
            raise TypeError("Invalid argument type for field reassignment: ", type(other))

    def set_dipole_type(self, my_type: str):
        assert my_type in ["Magnetic", "Electric"], f"""type must be either Magnetic or Electric, you selected {my_type}"""
        for d in self.dipoles:
            d.type = my_type
        return self
    
    def get_moments_on_grid(self, f_index=0, regular_grid=False, return_grid=False)-> Tuple[np.ndarray, np.ndarray]:
        M = self.M[:, f_index]
        if regular_grid:
            return_moments_on_expanded_grid(M, self.r0)
        return return_moments_on_grid(M, self.r0)
        
    def get_single_type_array(self, my_type: str)-> 'DipoleSourceArray':
        my_type_dipole_list = [d for d in self.dipoles if d.type == my_type]
        return DipoleSourceArray.init_dipole_array_from_dipole_list(self.f, my_type_dipole_list)

    
    def decompose_array(self):
        dec_dipoles = [d.decompose_dipole() for d in self.dipoles]
        empty_dipole_source_array = DipoleSourceArray.init_dipole_array_from_dipole_list(self.f, [])
        if len(dec_dipoles) == 0:
            return empty_dipole_source_array, empty_dipole_source_array, empty_dipole_source_array
        d_x, d_y, d_z = np.array(dec_dipoles).T

        darray_x = DipoleSourceArray.init_dipole_array_from_dipole_list(self.f, list(filter(lambda x: x!=0, d_x)))
        darray_y = DipoleSourceArray.init_dipole_array_from_dipole_list(self.f, list(filter(lambda x: x!=0, d_y)))
        darray_z = DipoleSourceArray.init_dipole_array_from_dipole_list(self.f, list(filter(lambda x: x!=0, d_z)))
        
        return darray_x, darray_y, darray_z
    
    def self_update_to_decomposed(self):
        d_x, d_y, d_z = self.decompose_array()
        new_list_dipoles = d_x.dipoles + d_y.dipoles + d_z.dipoles
        return self.init_dipole_array_from_dipole_list(self.f, new_list_dipoles)
    
    def get_list_of_dipoles(self) -> Iterable[dipole_source.Dipole_Source]:
        dipoles = []
        
        if self.f.shape == (1,) and np.ndim(self.dipole_moments) == 1:
            self.dipole_moments = self.dipole_moments[:, None]

        for ii in range(self.N_dipoles):
            dipoles.append(
                dipole_source.Dipole_Source(
                    self.r0[ii], 
                    orientation=self.orientations[ii], 
                    source= (signals.Frequency_Signal_Base(self.dipole_moments[ii], f=self.f) if self.dipole_moments is not None else None),
                    domain="Frequency"
                )
            )
        self.dipoles = dipoles
        return dipoles

    
    def get_source_matrices(self):
        positions = np.array([loop.r0 for loop in self.dipoles])
        moment_matrices = np.array([loop.get_overall_source_matrix() for loop in self.dipoles])

        # moment shape should be (N_dipoles, 3, N_frequencies)

        return positions, moment_matrices
    
    def set_individual_dipole_type(self, types_list):
        assert len(types_list)==self.N_dipoles

        valid_values = set(["Electric", "Magnetic", None])
        assert set(types_list).issubset(valid_values), "all elements must be either \"Electric\" or \"Magnetic\" types"

        for d, tp in zip(self.dipoles, types_list):
            d.type = tp
        return self
    
    def get_single_test_dipole_on_array(self, orientation: str, r0: np.ndarray=None, f: np.ndarray=None, type: str="Electric"):
        my_dict = dict(x=[np.pi/2,0], y=[np.pi/2, np.pi/2], z=[0,0])
        if r0 is None:
            r0 = [0,0,0]
        if len(r0) == 2:
            r0 = [r0[0], r0[1], 0]

        my_dipole = dipole_source.Dipole_Source(
            r0 if r0 is not None else [0,0,0], 
            orientation=my_dict[orientation],  
            source=signals.Frequency_Signal_Base([1], f=f if f is not None else self.f), 
            domain="Frequency",
            type=type
            )
        return self.init_dipole_array_from_dipole_list(self.f, [my_dipole])


    @staticmethod
    def init_dipole_array_from_dipole_list(f, dipoles):

        positions = [dipole.r0 for dipole in dipoles] 
        orientations = [[dipole.theta, dipole.phi] for dipole in dipoles]
        moments = [dipole.dipole_moment.signal for dipole in dipoles]
        type_list = [d.type for d in dipoles]

        return DipoleSourceArray(f, positions, orientations, moments).set_individual_dipole_type(type_list)


class FlatDipoleArray(DipoleSourceArray):
    def __init__(self, f: Iterable, height:int, r0: Iterable, orientations: Union[Iterable, str], moments=None, type=None): 

        if len(r0) != 0:
            if r0.shape[1] == 3:
                r0 = r0[:,:-1] # remove the last column               
            r03 = np.concatenate([r0, np.full_like(r0[:,0,None], height, dtype=np.float64)], axis=1)

        else: 
            r03 = []
        super().__init__(f, r03, orientations, moments, type)
        self.height = height
        self.r02 = r0

    def __floordiv__(self, other) -> "FlatDipoleArray":
        source_array = super().__floordiv__(other)
        return FlatDipoleArray.init_from_array(source_array, self.height)
    

    def __add__(self, other):
        result = super().__add__(other)
        heights = [d.r0[2] for d in result.dipoles]

        if len(heights) == 0:
            return FlatDipoleArray.init_from_array(result, self.height)
        
        if len(set(heights)) == 1:
            return FlatDipoleArray.init_from_array(result, heights[0])

        # check if heights are all the same value
        if len(set(heights)) != 1:
            print(f"not all dipoles have the same height, forcing to {self.height}")
            return FlatDipoleArray.init_from_array(result, self.height)
    
    def __sub__(self, other):
        result = super().__sub__(other)
        return FlatDipoleArray.init_from_array(result, self.height)
    
    def get_single_test_dipole_on_array(self, orientation: str, r0=None, *args, **kargs)->"FlatDipoleArray":
        if r0 is None:
            r0 = [0,0,self.height]
        if len(r0)==2:
            r0 = [r0[0], r0[1], self.height]

        return FlatDipoleArray.init_from_array(super().get_single_test_dipole_on_array(orientation, r0=r0, *args, **kargs))
    
    def create_empty_array(self):
        return FlatDipoleArray.init_dipole_array_from_dipole_list(self.f, [])

    @staticmethod
    def init_dipole_array_from_dipole_list(f, dipoles):
        result = DipoleSourceArray.init_dipole_array_from_dipole_list(f, dipoles)
        return FlatDipoleArray.init_from_array(result)
    
    @staticmethod
    def init_from_cell_array(cell_arr: CellArray, height, orientations, f, moments=None, type=None):
        r0 = cell_arr.center_points.reshape(2,-1).T
        return FlatDipoleArray(f, height, r0, orientations, moments, type)

    @staticmethod
    def init_from_array(d_array:DipoleSourceArray, height=None):
        r03 = d_array.r0
        if d_array.N_dipoles>0:
            height = r03[0,-1] if height is None else height
        else:
            height = 0
        
        ors = d_array.orientations
        f = d_array.f
        moments = d_array.dipole_moments
        
        type_list = [d.type for d in d_array.dipoles]
        return FlatDipoleArray(f, height, r03, ors, moments).set_individual_dipole_type(type_list)
        
        
    
        
        

def combine_arrays(array1:DipoleSourceArray, array2:DipoleSourceArray):
    """for now only works with arrays with the same frequencies"""
    
    dipoles = array1.dipoles + array2.dipoles
    combined_array = DipoleSourceArray.init_dipole_array_from_dipole_list(array1.f, dipoles)

    return combined_array

def fix_alligned_dipole_orientation(n_dipoles, theta, phi)->np.ndarray:
    theta = np.full(n_dipoles, theta)
    phi = np.full(n_dipoles, phi)

    return np.stack([theta, phi], axis=1)