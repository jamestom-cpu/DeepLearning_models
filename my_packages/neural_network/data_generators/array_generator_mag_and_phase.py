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
# import matplotlib.colors as mcolors



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
    
    def plot_labeled_data(self, fields, targets, index=0, mask_padding=0, ax=None, FIGSIZE=(15,3), image_folder="/workspace/images", savename="temp.png"):
        if np.ndim(targets)==4: 
            targets = targets[0]
        return super().plot_labeled_data(fields, targets, index, mask_padding, ax, FIGSIZE, image_folder, savename) 

    def plot(self, fields, targets, index=None):
        fig, ax = plt.subplots(3, 3, figsize=(15,5), constrained_layout=True)
        self.plot_labeled_data(fields, targets[0], ax=ax[0], index=index)
        self.plot_target_magnitude(targets, ax=ax[1])
        self.plot_target_phase(targets, ax=ax[2])
        return fig, ax

    def plot_target_magnitude(self, targets, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1,3, figsize=(10,3), constrained_layout=True)
        else:
            fig = ax[0].get_figure()
        abs_moments = targets[1]
        grid_x, grid_y = self.r0_grid[:2, ..., 0]

        titles = ["Ez", "Hx", "Hy"]
        labels = ["|E|", "|H|", "|H|"]
        formatter = plt.FuncFormatter(lambda ms, x: "{:.1f}".format(ms*1e3))
        for ii in range(3):
            q = ax[ii].pcolormesh(grid_x, grid_y, abs_moments[ii], cmap="RdBu_r")
            ax[ii].set_title(titles[ii])
            ax[ii].set_xlabel("x (mm)")
            ax[ii].set_ylabel("y (mm)")
            ax[ii].xaxis.set_major_formatter(formatter)
            ax[ii].yaxis.set_major_formatter(formatter)
            # create a colorbar for the magnitudes
            cbar = fig.colorbar(q, ax=ax[ii], label=labels[ii])
            # Add a tick at the maximum value of the field
            max_value = np.max(abs_moments[ii])
            min_value = np.min(abs_moments[ii])
            new_ticks = np.linspace(min_value, max_value, 5)
            cbar.set_ticks(new_ticks)
            cbar.set_ticklabels(["{:.2e}".format(tick) for tick in new_ticks])
        
    def plot_target_phase(self, targets, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
        else:
            fig = ax[0].get_figure()
        phase_values = targets[2]
        grid_x, grid_y = self.r0_grid[:2, ..., 0]

        titles = ["Ez Phase", "Hx Phase", "Hy Phase"]
        for ii in range(3):
            q = ax[ii].pcolormesh(grid_x, grid_y, phase_values[ii], cmap="twilight")
            ax[ii].set_title(titles[ii])
            ax[ii].set_xlabel("x (mm)")
            ax[ii].set_ylabel("y (mm)")
            cbar = fig.colorbar(q, ax=ax[ii], label="Phase (rad)")
            cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            cbar.set_ticklabels(["$-\pi$", "$-\pi/2$", "0", "$\pi/2$", "$\pi$"])

        

        



