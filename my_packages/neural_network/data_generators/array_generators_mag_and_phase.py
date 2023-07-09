import numpy as np
from typing import Iterable, Tuple
import os, sys
import torch.nn.functional as F
import torch

main_workspace_path = "/workspace"
sys.path.append(main_workspace_path)


from my_packages.classes.aux_classes import Grid
from my_packages.classes.dipole_array import FlatDipoleArray
from my_packages.classes.dipole_fields import DFHandler_over_Substrate
from my_packages.classes.model_components import UniformEMSpace, Substrate
from my_packages.classes.field_classes import Scan

from my_packages.neural_network.data_generators.magnetic_array_generator import RandomMagneticDipoleGenerator
from my_packages.neural_network.data_generators.electric_array_generator import RandomElectricDipoleGenerator
from my_packages.neural_network.data_generators.abstract import Generator
from my_packages.neural_network.data_generators.mixed_array_generator import MixedArrayGenerator

import matplotlib.pyplot as plt




class ArrayGenerator_MagnitudesAndPhases(MixedArrayGenerator):

    def _return_target_moments_and_phases(self, complex_moments):
        M = np.abs(complex_moments); ph = np.angle(complex_moments)
        return M, ph
    
    def return_moments_and_phases(self):
        mom_h = self.dfh.magnetic_array.M
        mom_e = self.dfh.electric_array.M

        M_e, ph_e = self._return_target_moments_and_phases(mom_e)
        M_h, ph_h = self._return_target_moments_and_phases(mom_h)
        
        Me_target = self.electric_generator._moments_on_grid(M_e)
        phase_e_target = self.electric_generator._moments_on_grid(ph_e)

        Mh_target = self.magnetic_generator._moments_on_grid(M_h)
        phase_h_target = self.magnetic_generator._moments_on_grid(ph_h)

        return Me_target, Mh_target, phase_e_target, phase_h_target
    
    def save_target_magnitudes_and_phases(self):
        self.Me, self.Mh, self.ph_e, self.ph_h = self.return_moments_and_phases()

    @property
    def target_M(self):
        return np.concatenate((self.Me, self.Mh), axis=0)
    
    @property
    def target_phases(self):
        return np.concatenate((self.ph_e, self.ph_h), axis=0)

    def generate_labeled_data(self, index=None):
        Ez, Hx, Hy = self.generate_random_fields()
        self.save_target_magnitudes_and_phases()    
        
        Ez = self._list_of_scans_to_list_of_numpys(Ez)
        Hx = self._list_of_scans_to_list_of_numpys(Hx)
        Hy = self._list_of_scans_to_list_of_numpys(Hy)

        fields = np.stack((Ez, Hx, Hy), axis=0)
        if index is not None:
            fields = fields[:, index, ...]

        targets = np.stack([self.mask, self.target_M, self.target_phases], axis=0)
        return fields, targets