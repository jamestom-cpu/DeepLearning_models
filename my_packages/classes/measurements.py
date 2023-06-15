import numpy as np
import pandas as pd
import seaborn as sns
from scipy import interpolate
from matplotlib import pyplot as plt
import copy

from my_packages.classes import field_classes
from my_packages import my_hdf5




class View_Measurement():
    def __init__(self, h5path):
        self.h5path = h5path
    
    def get_fixture_fields(self, group):
        E3d, H3d, grid, f_solutions = my_hdf5.get_fields_from_h5(self.h5path, group, get_freqs=True, show_print=False)
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