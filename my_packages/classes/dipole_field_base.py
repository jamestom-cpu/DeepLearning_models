import numpy as np
from typing import Iterable, Tuple
from copy import copy
from typing import Union
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap






from my_packages import spherical_coordinates, field_calculators
from . import green_sols, dipole_array as d_array
from .dipole_source import Dipole_Source
from .dipole_array import DipoleSourceArray
from .model_components import UniformEMSpace
from .aux_classes import Grid
from my_packages.auxillary_funcs_moments import return_moments_on_grid, return_moments_on_expanded_grid



class Dipole_Field_Base():
    def __init__(self, EM_space: UniformEMSpace, dipole_array=DipoleSourceArray
                ):

        # eps_r is set to the value for air == 1.006
        # sigma of air is estimated around 1.2e-12
        eps_r = EM_space.eps_r
        mu_r = EM_space.mu_r
        sigma = EM_space.sigma
        r = EM_space.r

        self.EM_space = EM_space
        self.c0 = 299792458.0278641
        self.eps0 = 8.854187812e-12
        self.mu0 = 4*np.pi*1e-7
        self.eps_r = eps_r
        self.eps = eps_r*self.eps0
        self.mu_r = mu_r
        self.sigma = sigma
        self.r = r
        self.spherical_coord_points = spherical_coordinates.to_spherical_grid(self.r)
        self._dipole_array = dipole_array
        self._wavelength = None
        self._wave_speed=None
        self._set_frequency_array(dipole_array.f)
    
    @property 
    def green_solutions(self):
        return green_sols.GreenSolutions(self)


    @property
    def dipole_array(self):
        return self._dipole_array
    
    @dipole_array.setter
    def dipole_array(self, dipole_array: Union[d_array.DipoleSourceArray, d_array.FlatDipoleArray]):
        self._dipole_array = dipole_array
        self._set_frequency_array(dipole_array.f)
        

    @property 
    def intrinsic_impedance(self):
        return self._intrinsic_impedance
    @intrinsic_impedance.setter
    def intrinsic_impedance(self):
        return

    @property
    def wavelength(self):
        return self._wavelength
    
    @wavelength.setter
    def wavelength(self):
        return 

    @property
    def wave_speed(self):
        return self._wave_speed
    
    @wave_speed.setter
    def wave_speed(self):
        return 

    @property 
    def eps_c(self):
        return self._eps_c
    @eps_c.setter
    def eps_c(self):
        return

    @property 
    def eps_cr(self):
        return self._eps_cr
    @eps_cr.setter
    def eps_cr(self):
        return
    
    @property
    def f(self):
        return self._f
    
    @f.setter
    def f(self, f):
        print("cannot directly set f!")
        print("change the values in the dipole array")
 
    def _set_frequency_array(self,f):
        f = np.array(f)
        self._f = f
        self._eps_c = self.eps*(1 - 1j*(self.sigma/(2*np.pi*f*self.eps)))
        self._eps_cr = self._eps_c/self.eps0
        self._intrinsic_impedance=np.sqrt(self.mu0*self.mu_r/self._eps_c)
        self._k = 2*np.pi*f*np.sqrt(self._eps_c*self.mu0)
        self._wave_speed = self.c0/np.sqrt(self.mu_r*self.eps_r)
        self._wavelength = self._wave_speed/f
        
    @property 
    def k(self):
        return self._k
    @k.setter
    def k(self,k):
        print("you cannot directly set the wave number!")
        print("set the frequency instead")
        return
    
    @property 
    def r(self):
        return self.EM_space.r

    @r.setter
    def r(self, R):
        R = Grid(R)
        if len(R.shape) < 4:
            raise Exception("grid must be 3D grid")
        
        self.EM_space.r = R


    

class Dipole_Field(Dipole_Field_Base):
    def __init__(self, EM_space: UniformEMSpace, dipole_array=DipoleSourceArray):
        super().__init__(EM_space, dipole_array)

    def __floordiv__(self, other):
        """create a shortcut to create a class with same properties, but different field values"""
        if isinstance(other, DipoleSourceArray):
            new_dfh = copy(self)
            new_dfh.dipole_array=other
            return new_dfh
        else:
            raise TypeError("Invalid argument type for field reassignment: ", type(other))


    def get_dipole_source_matrices(self, dipoles: Iterable[Dipole_Source])-> Tuple[np.ndarray, np.ndarray]:
        positions = np.array([dd.r0 for dd in dipoles])
        moments = np.array([np.outer(dd.unit_p, dd.dipole_moment.signal) for dd in dipoles])

        # moment shape should be (N_dipoles, 3(x,y,z), N_frequencies)
        return positions, moments

        
    # def green_dyadic_solution(self, r0=0, scale_with_wave_number = True)-> np.ndarray:
    #     return  field_calculators.magnetic_greens_functions_solution_dyad(self.k, self.r, r0, scale_with_wave_number)

    # def green_for_single_dipole_amplitude(self, r0, orientations, scale_with_wave_number=True)-> np.ndarray:
    #     return field_calculators.magnetic_greens_solution_for_moment_amplitude(self.k, self.r, r0, orientations, scale_with_wave_number)
        
    # def green_dyadic_solution_array(self, scale_with_wave_number=True)-> np.ndarray:
    #     green_sols = [self.green_dyadic_solution(dipole.r0, scale_with_wave_number) for dipole in self.dipole_array.dipoles]
    #     return np.stack(green_sols, axis=0)
    
    # def green_dyadic_solution_array_for_moment_amplitude(self, scale_with_wave_number=True)-> np.ndarray:
    #     green_sols = [self.green_for_single_dipole_amplitude(dipole.r0, np.array([dipole.theta, dipole.phi]), scale_with_wave_number) for dipole in self.dipole_array.dipoles]
    #     return np.stack(green_sols, axis=0)
    
    
    # def calculate_field(self) -> np.ndarray:
    #     _, moments = self.dipole_array.get_source_matrices()
    #     GREEN = self.green_dyadic_solution_array()

    #     dipole_field = np.einsum("ijk, ij...k->...k", moments, GREEN)
    #     return dipole_field

    
    def get_centered_field_spherical_coords(self)-> np.ndarray:
        spherical_coord_centered_field = field_calculators.centered_field_in_polar_coords(self.spherical_coord_points, self.k)
        return spherical_coord_centered_field
    
    def return_moments_on_r_grid(self, f_index=0):
        M, G = self.dipole_array.get_moments_on_grid(f_index=f_index, return_grid=True)
        return return_moments_on_expanded_grid(M, G, self.r)


    
    def _plot_moment_func(self, func_on_M:Iterable, ax=None, expanded_grid=True, f_index=0, colorbar_label=None, **kwargs):
        
        if expanded_grid:
            M, grid_M = self.return_moments_on_r_grid()
        else:
            print("plotting on restricted grid")
            M, grid_M = self.dipole_array.get_moments_on_grid(f_index=f_index, return_grid=True)
        (xmin, xmax), (ymin, ymax), _= self.r.bounds()
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        im = ax.pcolormesh(*grid_M[:2, ..., 0]*1e3, func_on_M(M), **kwargs)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_xlim(xmin*1e3, xmax*1e3)
        ax.set_ylim(ymin*1e3, ymax*1e3)
        ax.set_facecolor("black")
        fig.colorbar(im, ax=ax, label=colorbar_label)

        
        return fig, ax, im

    def plot_moment_intensity(self, ax=None, expanded_grid=True, f_index=0, title=None, **kwargs):
        # find units
        dipole_types = set([dd.type for dd in self.dipole_array.dipoles])

        if len(dipole_types) > 1:
            units = "."
        elif "Electric" in dipole_types:
            units = "Vm"
        elif "Magnetic" in dipole_types:
            units = "Am"
        
        kwargs = {"colorbar_label": units} | kwargs
       
        fig, ax, im = self._plot_moment_func(
            func_on_M=np.abs,
            ax=ax, expanded_grid = expanded_grid, 
            f_index = f_index,
            **kwargs)
        

        
        def_title = "Moment Intensity"
        if title is not None:
            title = def_title + " - " + title
        ax.set_title(title)
        return fig, ax
    def plot_moment_phase(self, ax=None, expanded_grid=True, f_index=0, title=None, cmap="hsv", **kwargs):
        units = "deg"

        # # define the custom colormap
        # cmap = getattr(plt.cm, cmap)
        # cmap_colors = cmap(np.arange(cmap.N))
        # cmap_colors[-1, :] = cmap_colors[0, :]  # set last color to first color
        # custom_cmap = ListedColormap(cmap_colors)

        kwargs = {"colorbar_label": units, "cmap":cmap, "vmin":-180, "vmax":180} | kwargs

        # function to plot the phase - chose with respect to the largest moment
        ii = np.argmax(np.abs(self.dipole_array.M[:, f_index]))
        def phase_func(x):
            angles = np.angle(x, deg=True) - np.angle(self.dipole_array.M[ii, f_index], deg=True)
            angles[angles > 180] -= 360
            angles[angles < -180] += 360
            return angles
            
        fig, ax, im = self._plot_moment_func(
            func_on_M=phase_func,
            ax=ax, expanded_grid = expanded_grid, 
            f_index = f_index, 
            **kwargs)
        # remove colorbar
        im.colorbar.remove()
        fig.colorbar(im, ax=ax, label=units, ticks=[-180, -90, 0, 90, 180])
        def_title = "Moment Phase"
        if title is not None:
            title = def_title + " - " + title
        ax.set_title(title)
        return fig, ax
    
    def inspect_moments(self, ax=None, expanded_grid=True, f_index=0, title=""):
        if ax is None:        
            fig, ax = plt.subplots(1,2, figsize=(12,4), constrained_layout=True)
        else:
            fig = ax[0].figure

        self.plot_moment_intensity(ax=ax[0], expanded_grid=expanded_grid, f_index=f_index, title=title)
        self.plot_moment_phase(ax=ax[1], expanded_grid=expanded_grid, f_index=f_index, title=title)
        
        my_title = title + " - Moment Intensity"
        ax[0].set_title(my_title)
        ax[0].set_facecolor("gray")

        my_title = title + " - Moment Phase"
        ax[1].set_title(my_title)
        ax[1].set_facecolor("gray")

        suptitle = "Dipole Moments Inspection"
        fig.suptitle(suptitle, weight="bold")

        return fig, ax