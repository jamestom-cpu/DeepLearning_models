from typing import List, Iterable, Self
from dataclasses import dataclass
import numpy as np
import os
import pickle

from .dipole_cell_grid import DipoleCellGrid, Mask

@dataclass
class DipoleArrayRef():
    freqs: Iterable[float]
    height: float


class DipoleLayoutPopulation():
    def __init__(
        self, 
        cell_grid: DipoleCellGrid, 
        dipole_array_ref: DipoleArrayRef,
        masks:List[Mask],
        dipole_type: str = "Magnetic"
        ):

    
        self.cell_grid = cell_grid
        self.masks = masks
        self.dipole_array_ref = dipole_array_ref
        self._dipole_type = dipole_type

        self.dipole_arrays = [cell_grid.update_mask(mask).ActiveArrays.all.generate_dipole_array(
            height=dipole_array_ref.height, 
            f= dipole_array_ref.freqs
        ).set_dipole_type(dipole_type) for mask in masks]
    
    @property
    def dipole_type(self):
        return self._dipole_type
    
    @dipole_type.setter
    def dipole_type(self, dipole_type: str):
        self._dipole_type = dipole_type
        self.dipole_arrays = [darray.set_dipole_type(dipole_type) for darray in self.dipole_arrays]
        
    
    def __len__(self):
        return len(self.masks)

        
    def update_population(self, new_masks: Mask)-> Self:
        return DipoleLayoutPopulation(self.cell_grid, self.dipole_array_ref, new_masks, self.dipole_type)
    
    def update_dipole_type(self, dipole_type: str)-> Self:
        self.dipole_arrays = [darray.set_dipole_type(dipole_type) for darray in self.dipole_arrays]
        self.dipole_type = dipole_type