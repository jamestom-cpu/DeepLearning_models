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
            field_res = (50,50),
            dipole_density = 0.5
            ) -> None:
        self.resolution = resolution
        self.dipole_density = dipole_density
        self.field_res = field_res
        self.xbounds = xbounds
        self.ybounds = ybounds
        self.dipole_height = dipole_height
        self.substrate_thickness = substrate_thickness
        self.substrate_epsilon_r = substrate_epsilon_r
        self.dynamic_range = dynamic_range 
        self.probe_height = probe_height 
        self.f = f
        self._generate_r()
        self._generate_r0_grid()

    def _generate_r(self):
        x = np.linspace(self.xbounds[0], self.xbounds[1], self.field_res[0])
        y = np.linspace(self.ybounds[0], self.ybounds[1], self.field_res[1])
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
    
    @abstractmethod
    def generate_labeled_data(self):
        pass

    @abstractmethod
    def plot_labeled_data(self):
        pass

    @abstractmethod
    def create_a_copy(self):
        pass