import numpy as np
import pandas as pd
import seaborn as sns
from scipy import interpolate
from matplotlib import pyplot as plt
import h5py
import copy

from my_packages.hdf5 import hd5f_utils
from my_packages.classes import field_classes

from IPython.utils import io


class View_Measurement():
    def __init__(self, h5path, group_name=None):
        self.h5path = h5path
        self.name = group_name
        if group_name is not None:
            self.get_fixture_fields(group_name)
            self.simulation_properties = get_group_attributes(self.h5path, group_name)
        
    
    def get_fixture_fields(self, group):
        E3d, H3d, grid, f_solutions = hd5f_utils.get_fields_from_h5(self.h5path, group, get_freqs=True, show_print=False)
        # transform grid to mm
        grid = np.array(grid)*1e-3
        self.E = field_classes.Field3D(E3d, f_solutions, grid)
        self.H = field_classes.Field3D(H3d, f_solutions, grid)
        return self

    def apply_random_measurement(self, random_measurement):
        self.Random = random_measurement
        return self

    def get_field_at_probe_height(self, field="E"):
        my_dict = dict(E=self.E, H=self.H)
        z_index = np.argmin(np.abs(self.Random.center_point[-1]-my_dict[field].grid[-1, 0, 0, :]))
        Field_2d = my_dict[field].get_2D_field("z", z_index)

        my_dict[field].at_probe_height = Field_2d
        return Field_2d
    
    def plot_measurements_on_field(self, f_index=0, field="E", trace_width=1, **kargs):
        my_dict = dict(E=self.E, H=self.H)
        field2d = self.get_field_at_probe_height(field)
        z_field = field2d.field[-1,...]
        table = pd.DataFrame(
            np.abs(z_field[..., f_index]), 
            index=pd.Index(my_dict[field].grid[0, :, 0, 0].round(2), name="X"),
            columns=pd.Index(my_dict[field].grid[1, 0, :, 0].round(2), name="Y")
            )
        zoom_border = trace_width*1.2/2
        cut_table = table.loc[np.abs(table.index)<zoom_border].transpose().loc[np.abs(table.index)<zoom_border].transpose()
        
        fig, ax = plt.subplots(1,2, figsize=(14,4))
        self.__plot_measurements_on_field(table.T, ax[0], fig, trace_width=trace_width, **kargs)
        self.__plot_measurements_on_field(cut_table.T, ax[1], fig, trace_width=trace_width, **kargs)
        
        return table
        
        

    def __plot_measurements_on_field(self, table, ax, fig, trace_width=1, units="", **kargs):
        

        random_x, random_y, random_z = self.Random.measurement_points.T
        
        im = ax.pcolormesh(table.index, table.columns, table, cmap="jet", alpha=1, **kargs)
        ax.scatter(random_x, random_y, marker="x", s=10, lw=0.7, c="k", alpha=0.35)
        ax.set_xlabel("X[mm]")
        ax.set_ylabel("Y[mm]")

        #center point
        ax.scatter(0,0, marker='X', color = 'r', s=20)

        # ax.set_xlim([-1,1])
        # ax.set_ylim([-1,1])
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(units)
        
        if trace_width is not None:
            ax.axhline(y=-trace_width/2, color = 'k', lw=2.5, alpha=0.3)
            ax.axhline(y=+trace_width/2, color = 'k', lw=2.5, alpha=0.3)

        x_std = self.Random.x_std 
        y_std = self.Random.y_std 
        #show standard deviations
        ax.axvline(x=x_std, color = 'b', linestyle=(0,(1,10)))
        ax.axvline(x=-x_std, color = 'b', linestyle=(0,(1,10)))
        ax.axvline(x=x_std*2, color = 'b', linestyle=(0,(1,10)))
        ax.axvline(x=-x_std*2, color = 'b', linestyle=(0,(1,10)))

        ax.axhline(y=y_std, color = 'b', linestyle=(0,(1,10)))
        ax.axhline(y=-y_std, color = 'b', linestyle=(0,(1,10)))
        ax.axhline(y=y_std*2, color = 'b', linestyle=(0,(1,10)))
        ax.axhline(y=-y_std*2, color = 'b', linestyle=(0,(1,10)))

        #center point
        ax.scatter(0,0, marker='X', color = 'r', s=20)



class RandomMeasurements():
    def __init__(
        self, 
        center_point=(0,0,1e-3), 
        y_std = 0.15*1e-3, 
        x_std = 0.15*1e-3, 
        z_std = 0,
        number_of_points = 10000,
        seed = None
        ):

        self.center_point = center_point
        self.y_std = y_std
        self.x_std = x_std
        self.z_std = z_std
        self.N = number_of_points
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        covariance_matrix = [[self.x_std**2, 0, 0],[0, self.y_std**2, 0], [0,0,self.z_std**2]]
        measurement_points = np.random.multivariate_normal(mean=self.center_point, cov = covariance_matrix, size=(self.N))

        self.measurement_points = measurement_points
    
        
def get_group_attributes(h5path, group_name):
    with h5py.File(h5path) as f:
        group = f[group_name]
        attributes = dict(group.attrs)
        return attributes
         

