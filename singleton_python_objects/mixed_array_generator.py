import os, sys
import h5py
import numpy as np
import pandas as pd
import scipy
import math as m
import cmath
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from pprint import pprint


plt.switch_backend('TkAgg')
PROJECT_CWD = r"/workspace/"
sys.path.append(PROJECT_CWD)

os.chdir(PROJECT_CWD)

from my_packages.neural_network.data_generators.mixed_array_generator import MixedArrayGenerator


# inspect the data
# data parameters
resolution=(11,11)
field_res = (30,30)
xbounds = [-1e-2, 1e-2]
ybounds = [-1e-2, 1e-2]
padding = None
dipole_height = 1e-3
substrate_thickness = 1.4e-2
substrate_epsilon_r = 4.7
dynamic_range = 2
probe_heights = [6e-3, 8e-3, 1e-2]
dipole_density_E = 0.1
dipole_density_H = 0.1
inclde_dipole_position_uncertainty = False


rmg = MixedArrayGenerator(
    resolution=resolution,
    xbounds=xbounds,
    ybounds=ybounds,
    padding=padding,
    dipole_height=dipole_height,
    substrate_thickness=substrate_thickness,
    substrate_epsilon_r=substrate_epsilon_r,
    probe_height=probe_heights,
    dynamic_range=dynamic_range,
    f=[1e9],
    field_res=field_res,
    dipole_density_E=dipole_density_E,
    dipole_density_H=dipole_density_H,
    include_dipole_position_uncertainty=inclde_dipole_position_uncertainty,
    )

# allow the import of the specific instance of the generator
def get_mixed_array_generator():
    return rmg