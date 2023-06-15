import numpy as np
import pandas as pd
import scipy
import math as m
import cmath
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import h5py
import os, sys

from mpl_toolkits.mplot3d import Axes3D
from pprint import pprint



PROJECT_CWD = r"/workspace"

print(os.getcwd())
os.chdir(PROJECT_CWD)
print(os.getcwd())

from my_packages.classes.aux_classes import Grid
from my_packages.classes.dipole_array import FlatDipoleArray

# dipoles should be places on a plane with a normal vector pointing in the z-direction
# start by placing the dipoles on a regular grid

# define the grid
xbounds = [-0.01, 0.01]
ybounds = [-0.01, 0.01]
dipole_height = 0. # we can place the dipoles on the xy-plane - the height of the dipoles
# can be considered in the inspection height.

grid_resolution = 50

grid = np.mgrid[
    xbounds[0]:xbounds[1]:grid_resolution*1j, 
    ybounds[0]:ybounds[1]:grid_resolution*1j, 
    dipole_height:dipole_height:1j]
grid = Grid(grid)


from my_packages.classes.dipole_array import FlatDipoleArray
from my_packages.classes.dipole_fields import DFHandler_over_Substrate
from my_packages.classes.model_components import UniformEMSpace, Substrate

resolution= 21
def generate_random_mask(n, m):
    mask = np.random.randint(2, size=(n, m))
    return mask

def generate_random_orientation_xy(num_dipoles):
    orientation_mask = np.random.randint(2, size=(num_dipoles))
    new_orientation_mask = np.zeros((num_dipoles, 2), dtype=float)

    new_orientation_mask[orientation_mask == 0] = [0, np.pi/2]
    new_orientation_mask[orientation_mask == 1] = [np.pi/2, np.pi/2]
    return new_orientation_mask

def generate_random_moments(num_dipoles):
    moments_r, moments_i = np.random.uniform(0.2, 1, size=(2, num_dipoles))
    return moments_r+1j*moments_i

def from_dense_grid_to_centercell_grid(grid, cellgrid_shape):
    """
    grid: Grid object
    cellgrid_shape: tuple of (n, m) where n and m are integers
    """
    xbounds, ybounds = grid.bounds()[:-1]
    x = np.linspace(xbounds[0], xbounds[1], cellgrid_shape[0]+1) 
    x = np.diff(x)/2 + x[:-1]

    y = np.linspace(ybounds[0], ybounds[1], cellgrid_shape[1]+1)
    y = np.diff(y)/2 + y[:-1]

    return Grid(np.meshgrid(x, y, grid.z, indexing="ij"))


mask = generate_random_mask(resolution, resolution)

def from_mask_to_dipole_positions2D(mask, grid):
    """
    mask: 2D array of 0's and 1's
    grid: Grid object
    """
    xbounds, ybounds, _ = grid.bounds()
    x = np.linspace(xbounds[0], xbounds[1], mask.shape[0])
    y = np.linspace(ybounds[0], ybounds[1], mask.shape[1])
    x, y = np.meshgrid(x, y)
    x = x[mask == 1]
    y = y[mask == 1]
    z = np.zeros_like(x)
    return np.vstack((x, y, z)).T



def from_mask_to_dipole_array():
    pass

r0_grid = from_dense_grid_to_centercell_grid(grid, mask.shape)
r0 = from_mask_to_dipole_positions2D(mask, r0_grid)
N_dipoles = len(r0)
ors = generate_random_orientation_xy(N_dipoles)
moments = generate_random_moments(N_dipoles)
f = [1e9]
dipole_height = 3e-3

# measurement points
probe_heights = 1e-7, 1.2e-2
substrate_thickness = 1.4e-2
x = np.linspace(xbounds[0]*1.2, xbounds[1]*1.2, grid_resolution)
y = np.linspace(ybounds[0]*1.2, ybounds[1]*1.2, grid_resolution)
z = list(probe_heights)
r = np.meshgrid(x, y, z, indexing="ij")
r = Grid(r)

dipole_array = FlatDipoleArray(f=f, height=dipole_height, r0=r0, orientations=ors, moments=np.expand_dims(moments, axis=-1), type="Magnetic")
EM_space = UniformEMSpace(r=r)
substrate = Substrate(x_size=0.01, y_size=0.01, thickness=substrate_thickness, material_name="FR4_epoxy")


dfh = DFHandler_over_Substrate(EM_space=EM_space, substrate=substrate, dipole_array=dipole_array)


fig, ax = plt.subplots()

dfh.evaluate_fields().H.run_scan("y", index = 0.7e-2).plot(ax=ax)
dfh.dh_magnetic.plot_moment_intensity()