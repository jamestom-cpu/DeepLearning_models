import numpy as np

import attr
from typing import Iterable, Union
from dataclasses import dataclass
from my_packages.classes.aux_classes import Grid

@dataclass
class Substrate():
    x_size: float
    y_size: float
    thickness: float
    material_name: str
    eps_r: float = 1 
    mu_r: float = 1

@attr.s
class DipoleArr_Prot:
    r0: Iterable = attr.ib()
    orientations: Iterable = attr.ib()
    dipole_moments: Iterable = attr.ib()
    dipoles: list = attr.ib()


@attr.s
class DipoleFC_Prot:
    k: Union[float, Iterable] = attr.ib()
    r: Iterable = attr.ib()
    dipole_array: DipoleArr_Prot
    f: Iterable
    eps_r: float
    eps0: float
    mu_r: float
    sigma: float

@attr.s
class MagneticFC_Prot(DipoleFC_Prot):
    k: Union[float, Iterable] = attr.ib()
    r: Iterable = attr.ib()
    dipole_array: DipoleArr_Prot = attr.ib()
    f: Iterable = attr.ib()
    eps_r: float = attr.ib()
    eps0: float = attr.ib()
    mu_r: float = attr.ib()
    mu0:float = attr.ib()
    substrate: Substrate = attr.ib()

    def keep_valid_dipoles(self):
        ...
    
    def get_image_positions(self, ground_plane_height: float) -> np.ndarray:
        ...
    
    def get_image_orientations(self)-> np.ndarray:
        ...


attr.s
class ElectricSubstrFC_Prot(DipoleFC_Prot):
    substrate: Substrate = attr.ib()

    def keep_valid_dipoles(self)-> np.ndarray:
        ...

    def get_image_orientations(self)->np.ndarray:
        ...
    
    def get_image_positions(self, ground_plane_height)-> np.ndarray:
        ...


