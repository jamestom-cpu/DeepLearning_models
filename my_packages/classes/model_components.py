import numpy as np
from .field_classes import Field3D

from my_packages.protocols import Substrate
from .aux_classes import Grid


class UniformEMSpace():
    def __init__(self, r, eps_r=1, mu_r=1, sigma=0):
        self.eps0 = 8.854187812e-12
        self.mu0 = 4*np.pi*1e-7
        self.eps_r = eps_r
        self.mu_r = mu_r
        self.sigma = sigma
        self.r = Grid(r)
       
    

