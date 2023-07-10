import numpy as np
from typing import Iterable, Tuple
import os, sys

main_workspace_path = "/workspace"
sys.path.append(main_workspace_path)


from my_packages.classes.aux_classes import Grid
from my_packages.classes.dipole_array import FlatDipoleArray
from my_packages.classes.dipole_fields import DFHandler_over_Substrate
from my_packages.classes.model_components import UniformEMSpace, Substrate
from my_packages.classes.field_classes import Scan


from abc import ABC, abstractmethod
   

class Generator(ABC):
    def __init__(
            self, resolution, 
            xbounds, ybounds, 
            dipole_height, 
            substrate_thickness,
            substrate_epsilon_r,
            probe_height,
            dynamic_range, f=[1e9],
            padding=None,
            field_res = (50,50),
            dipole_density = 0.5
            ) -> None:
        self.resolution = resolution
        self.dipole_density = dipole_density
        self.field_res = field_res
        self.xbounds = xbounds
        self.ybounds = ybounds
        self.padding = padding
        self.dipole_height = dipole_height
        self.substrate_thickness = substrate_thickness
        self.substrate_epsilon_r = substrate_epsilon_r
        self.dynamic_range = dynamic_range 
        self.probe_height = probe_height 
        self.f = f
        self._generate_r()
        self._generate_r0_grid()

    def _generate_r(self):
        if self.padding is None:
            xbounds = self.xbounds
            ybounds = self.ybounds
        else:
            xbounds = [self.xbounds[0]-self.padding[0], self.xbounds[1]+self.padding[0]]
            ybounds = [self.ybounds[0]-self.padding[1], self.ybounds[1]+self.padding[1]]


        x = np.linspace(xbounds[0], xbounds[1], self.field_res[0])
        y = np.linspace(ybounds[0], ybounds[1], self.field_res[1])
        if isinstance(self.probe_height, Iterable):
            z = self.probe_height
        else:
            z = [self.probe_height]
        self.r = Grid(np.meshgrid(x, y, z, indexing="ij"))

    def _generate_r0_grid(self):
        """
        grid: Grid object
        cellgrid_shape: tuple of (n, m) where n and m are integers
        """   

        x = np.linspace(self.xbounds[0], self.xbounds[1], self.resolution[0]+1) 
        x = np.diff(x)/2 + x[:-1]

        y = np.linspace(self.ybounds[0], self.ybounds[1], self.resolution[1]+1)
        y = np.diff(y)/2 + y[:-1]

        self.r0_grid = Grid(np.meshgrid(x, y, [self.dipole_height], indexing="ij"))
        return self.r0_grid
    
    def _generate_random_moments(self):
        moments_abs = np.random.uniform(1/self.dynamic_range, 1, size=(self.N_dipoles,))
        moments_phase = np.random.uniform(-np.pi, np.pi, size=(self.N_dipoles,))
        return moments_abs * np.exp(1j * moments_phase)
    
    def _moments_on_grid(self, mom):
        """mom can either be M or ph"""
        mask = self.mask
        # substitute the 1s of the mask with the values of mom
        mom_on_grid = np.zeros_like(mask, dtype=np.float64)
        mom_on_grid[mask == 1] = mom.flatten()
        return mom_on_grid
    
    
    @abstractmethod
    def generate_labeled_data(self):
        pass

    @abstractmethod
    def plot_labeled_data(self):
        pass

    @abstractmethod
    def create_a_copy(self):
        pass