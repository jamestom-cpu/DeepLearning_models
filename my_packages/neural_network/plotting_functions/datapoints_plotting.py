import os, sys
import numpy as np
import matplotlib.pyplot as plt

import torch

from .aux_datapoints_plotting import plotting_func, H_plotting_func, E_plotting_func, input_data_to_Scan
from my_packages.classes.aux_classes import Grid


class Plotter():
    def __init__(self, grid, label_grid = None, f=1e9, resolution = (11, 11)):
        self.grid = grid
        self.f = f

        if label_grid is None:
            self.label_grid = self._generate_r0_grid(resolution)
        else:
            self.label_grid = label_grid
    
    def _generate_r0_grid(self):
        """
        grid: Grid object
        cellgrid_shape: tuple of (n, m) where n and m are integers
        """   

        x = np.linspace(self.xbounds[0], self.xbounds[1], self.resolution[0]+1) 
        x = np.diff(x)/2 + x[:-1]

        y = np.linspace(self.ybounds[0], self.ybounds[1], self.resolution[1]+1)
        y = np.diff(y)/2 + y[:-1]

        return Grid(np.meshgrid(x, y, [self.dipole_height], indexing="ij"))
  
    

    def _plot_labeled_data(self, fields, label, index=0, ax=None, FIGSIZE=(15,3)):        
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label)
        # return to numpy
        label = label.numpy()
        
        Ez, Hx, Hy = input_data_to_Scan(fields, self.grid, freq=self.f)
        x, y = self.label_grid[:-1, ..., 0]
        plotting_func(Ez[index], Hx[index], Hy[index], x, y, label, ax=ax, FIGSIZE=FIGSIZE)
        
        

    def plot_Hlabeled_data(self, fields, label, index=0, ax=None, FIGSIZE=(15,3)):
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label)

        label = label.numpy()

        if np.ndim(fields) == 3:
            fields = np.expand_dims(fields, axis=1)

        _, Hx, Hy = input_data_to_Scan(fields, self.grid, freq=self.f)
        x, y = self.label_grid[:-1, ..., 0]

        H_plotting_func(Hx[index], Hy[index], x, y, label, ax=ax, FIGSIZE=FIGSIZE)


    def plot_Elabeled_data(self, fields, label, index=0, ax=None, FIGSIZE=(15,3)):     
        Ez, _, _ = input_data_to_Scan(fields, self.grid, freq=self.f)
        x, y = self.label_grid[:-1, ..., 0]
        E_plotting_func(Ez[index], x, y, label, ax=ax, FIGSIZE=FIGSIZE)


    def plot_labeled_data(self, fields, targets, index=0, ax=None, FIGSIZE=(15,3), image_folder="/workspace/images", savename="temp.png"):
            if np.ndim(targets)==4: 
                targets = targets[0]
            nfields = fields.shape[0]
            if nfields == 3:
                return self._plot_labeled_data(fields, targets, index, ax, FIGSIZE) 
            if nfields == 2: 
                return self.plot_Hlabeled_data(fields, targets, index, ax, FIGSIZE)

    def plot_all(self, fields, targets, index=None, ax=None, normalized_phase=False):
        nlayers = targets.shape[0]
        nfields = targets.shape[1]
        if ax is None:
            fig, ax = plt.subplots(nlayers, nfields, figsize=(15,5), constrained_layout=True) 
        else: 
            fig = ax.flatten()[0].get_figure()    
        self.plot_labeled_data(fields, targets[0], ax=ax[0], index=index)
        
        self.plot_target_magnitude(targets, ax=ax[1])
        self.plot_target_phase(targets, ax=ax[2:], normalized01=normalized_phase)
        return fig, ax


    def plot_target_magnitude(self, targets, ax=None):
        targets = np.asarray(targets)
        nfields = targets.shape[1]
        if ax is None:
            fig, ax = plt.subplots(1, nfields, figsize=(10,3), constrained_layout=True)
        else:
            fig = ax.flatten()[0].get_figure()
        abs_moments = targets[1]
        grid_x, grid_y = self.label_grid[:2, ..., 0]

        if nfields==3:
            titles = ["Ez", "Hx", "Hy"]
            labels = ["|E|", "|H|", "|H|"]
        elif nfields==2:
            titles = ["Hx", "Hy"]
            labels = ["|H|", "|H|"]
        formatter = plt.FuncFormatter(lambda ms, x: "{:.1f}".format(ms*1e3))
        for ii in range(nfields):
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

    def plot_expanded_phases(self, targets, ax=None):
        targets = np.asarray(targets)
        nfields = targets.shape[1]
        mask = targets[0]
        if ax is None:
            fig, ax = plt.subplots(2, nfields, figsize=(10, 3), constrained_layout=True)
        else:
            fig = ax.flatten()[0].get_figure()

        sin_phase_values, cos_phase_values = targets[-2:]
        sin_phase_values = np.where(mask==0, np.nan, sin_phase_values)  # Mask values
        cos_phase_values = np.where(mask==0, np.nan, cos_phase_values)  # Mask values

        grid_x, grid_y = self.label_grid[:2, ..., 0]
        if nfields==3:
            titles = ["Ez Phase", "Hx Phase", "Hy Phase"]
        elif nfields==2:
            titles = ["Hx Phase", "Hy Phase"]

        ticks = [-1, -0.5, 0, 0.5, 1]
        tick_labels = ["-1", "-0.5", "0", "0.5", "1"]
        vmin = -1 ; vmax = 1
        cmap = plt.cm.twilight
        cmap.set_bad('black', 0.7)  # set 'nan' values color to black

        for ii in range(0, nfields):
            q_sin = ax[0, ii].pcolormesh(grid_x, grid_y, sin_phase_values[ii], cmap=cmap, vmin=vmin, vmax=vmax)
            q_cos = ax[1, ii].pcolormesh(grid_x, grid_y, cos_phase_values[ii], cmap=cmap, vmin=vmin, vmax=vmax)
            ax[0, ii].set_title(titles[ii]+" sin")
            ax[1, ii].set_title(titles[ii]+" cos")
            cbar1 = fig.colorbar(q_sin, ax=ax[0, ii], label="Phase (rad)")
            cbar2 = fig.colorbar(q_cos, ax=ax[1, ii], label="Phase (rad)")
            cbar1.set_ticks(ticks)
            cbar1.set_ticklabels(tick_labels)
            cbar2.set_ticks(ticks)
            cbar2.set_ticklabels(tick_labels)

        for ii in range(2*nfields):
            ax.flatten()[ii].set_xlabel("x (mm)")
            ax.flatten()[ii].set_ylabel("y (mm)")
        
        
    
    def plot_target_phase(self, targets, ax=None, normalized01=False):
        targets = np.asarray(targets)

        if targets.shape[0]==4:
            return self.plot_expanded_phases(targets, ax=ax)

        nfields = targets.shape[1]
        if ax is None:
            fig, ax = plt.subplots(1, nfields, figsize=(10, 3), constrained_layout=True)
        else:
            fig = ax.flatten()[0].get_figure()
        phase_values = targets[2]
        grid_x, grid_y = self.label_grid[:2, ..., 0]
        if nfields==3:
            titles = ["Ez Phase", "Hx Phase", "Hy Phase"]
        elif nfields==2:
            titles = ["Hx Phase", "Hy Phase"]
        if normalized01:
            ticks = [0, 0.25, 0.5, 0.75, 1]
            tick_labels = ["0", "0.25", "0.5", "0.75", "1"]
            vmin = 0; vmax = 1
        else:
            ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
            tick_labels = ["$-\pi$", "$-\pi/2$", "0", "$\pi/2$", "$\pi$"]
            vmin = -np.pi; vmax = np.pi

        ax = ax.flatten()

        for ii in range(nfields):
            q = ax[ii].pcolormesh(grid_x, grid_y, phase_values[ii], cmap="twilight", vmin=vmin, vmax=vmax)
            ax[ii].set_title(titles[ii])
            ax[ii].set_xlabel("x (mm)")
            ax[ii].set_ylabel("y (mm)")
            cbar = fig.colorbar(q, ax=ax[ii], label="Phase (rad)")
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(tick_labels)

        