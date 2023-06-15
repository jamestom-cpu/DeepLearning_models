import numpy as np

from dataclasses import dataclass
from typing import Tuple

from my_packages.model import green_solution_methods
from my_packages.classes.field_classes import Field3D, PartialField, PartialField2D
from my_packages.classes.dipole_fields import DFHandler_over_Substrate

@dataclass
class OptimalSourcesValues():

    def __init__(
        self, fh : DFHandler_over_Substrate, 
        measured_E: Field3D,  
        measured_H: Field3D,
        reg_coefficient: float = 1e-7, 
        N_expansion: int = 15,
        Mhh: np.ndarray = None,
        Mee: np.ndarray = None,
        ):

        self._reconstructed_E: Field3D = measured_E//np.zeros_like(measured_E.field)
        self._reconstructed_H: Field3D = measured_H//np.zeros_like(measured_H.field)
        self.delta_E: Field3D = measured_E
        self.delta_H: Field3D = measured_H

        self.reg_coefficient = reg_coefficient
        self.fh= fh
        self.measured_E = measured_E
        self.measured_H = measured_H

        self.N_expansion= N_expansion
        self.Mhh = Mhh
        self.Mee = Mee

        

    @property
    def reconstructed_E(self):
        return self._reconstructed_E
    @property
    def reconstructed_H(self):
        return self._reconstructed_H

    @property
    def Ghh(self):
        return self.fh.dh_magnetic.green_solutions.Ghh(-self.fh.substrate.thickness)
    
    @property
    def Ghe(self):
        return self.fh.dh_magnetic.green_solutions.Ghe(-self.fh.substrate.thickness, self.N_expansion)

    @property
    def Gee(self):
        return self.fh.dh_electric.green_solutions.Gee(-self.fh.substrate.thickness, self.N_expansion)
    
    @property
    def Geh(self):
        return self.fh.dh_electric.green_solutions.Geh(-self.fh.substrate.thickness, self.N_expansion)

    def optimize_magnetic_dipoles(self, target_H: Field3D=None, verbose=False):
        def mprint(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)
                
        if target_H is None:
            target_H = self.measured_H

        if isinstance(target_H, (PartialField, PartialField2D)):
            mapping_dict = dict(x=0, y=1, z=2)
            indices = sorted([mapping_dict[component] for component in target_H.components])
            mprint("indices", indices)
            Ghh = self.Ghh[:, indices, ...]
            mprint("Ghh.shape", Ghh.shape)
        else:
            Ghh = self.Ghh
            mprint("Ghh.shape", Ghh.shape)
        # matrix pseudo inversion

        mprint("target_H.field.shape", target_H.field.shape)
        Mhh = green_solution_methods.solve_with_NormalEquations_matrix(
            G=Ghh, target_field=target_H.field, regularization_lambda=self.reg_coefficient)
        #update the magnetic array
        self.Mhh = Mhh
        self._update_magnetic_dipoles(Mhh)
        return self


    def optimize_electric_dipoles(self, target_E: Field3D=None, verbose=False):
        if target_E is None:
            target_E = self.measured_E
        
        if isinstance(target_E, PartialField):
            mapping_dict = dict(x=0, y=1, z=2)
            indices = sorted([mapping_dict[component] for component in target_E.components])
            Gee = self.Gee[:, indices, ...]
        else:
            Gee = self.Gee
        # matrix pseudo inversion
        Mee = green_solution_methods.solve_with_NormalEquations_matrix(
            G = Gee, target_field=target_E.field, regularization_lambda=self.reg_coefficient)
        self.Mee =Mee
        #update the magnetic array
        self._update_electric_dipoles(Mee)
        return self

    def _update_magnetic_dipoles(self, new_moments: np.ndarray):
        self.fh.magnetic_array = self.fh.magnetic_array//new_moments
        return self
    
    def _update_electric_dipoles(self, new_moments: np.ndarray):
        self.fh.electric_array = self.fh.electric_array//new_moments
        return self
    
    def evaluate_H(self):
        self._reconstructed_H = self.fh.dh_magnetic.evaluate_magnetic_field_over_PEC(-self.fh.substrate.thickness)
        return self
    
    def evaluate_E(self, N=None):
        if N is None:
            N = self.N_expansion
        self._reconstructed_E = self.fh.dh_electric.evaluate_electric_field_over_PEC(-self.fh.substrate.thickness, N=N)
        return self
    
    # def evaluate_E_approx(self):
    #     self._reconstructed_E = self.fh.dh_electric.evaluate_electric_field_over_PEC(-self.fh.substrate.thickness, N=1)
    #     return self
    
    def evaluate_mse_H(self):
        rec_H = self.reconstructed_H.normalized_field((self.measured_H.max, self.measured_H.min))
        targ_H = self.measured_H.normalized_field()
        return abs((targ_H-rec_H)**2).mean()
    
    def evaluate_mse_E(self):
        rec_E = self.reconstructed_E.normalized_field((self.measured_E.max, self.measured_E.min))
        targ_E = self.measured_E.normalized_field()
        return abs((targ_E-rec_E)**2).mean()

    # def update_delta_fields(self):       
    #     self.delta_E = self.measured_E - self.reconstructed_E
    #     self.delta_H = self.measured_H - self.reconstructed_H
    #     return self
    
    def evaluate_fields(self):
        self.fh.evaluate_fields(N=self.N_expansion)
        self._reconstructed_E, self._reconstructed_H = self.fh.E, self.fh.H
        return self    
    
    def run_iteration(self, reg_coeff_H=None, reg_coeff_E=None, verbose = False) -> DFHandler_over_Substrate:
        if reg_coeff_H is None:
            reg_coeff_H = self.reg_coefficient
        if reg_coeff_E is None:
            reg_coeff_E = self.reg_coefficient

        target_H = self.delta_H
        self.reg_coefficient = reg_coeff_H
        self.optimize_magnetic_dipoles(target_H=target_H, verbose=verbose).evaluate_fields().update_delta_fields()
        target_E = self.delta_E
        self.reg_coefficient = reg_coeff_E
        self.optimize_electric_dipoles(target_E=target_E, verbose=verbose).evaluate_fields().update_delta_fields()
        return self
    
    def update_delta_fields(self):
        self.evaluate_fields()
        self.delta_E = self.measured_E - self.fh.dh_magnetic.evaluate_electric_field_over_PEC(
            GND_h=-self.fh.substrate.thickness, N=self.N_expansion
            )
        self.delta_H = self.measured_H - self.fh.dh_electric.evaluate_magnetic_field_over_PEC(
            GND_h=-self.fh.substrate.thickness, N=self.N_expansion
        )
        return self
    

    def evaluate_mse(self):
        self.fh.evaluate_fields(N=self.N_expansion)
        E, H = self.fh.E, self.fh.H

        targ_E, targ_H = self.measured_E.normalized_field(), self.measured_H.normalized_field()
        rec_E, rec_H = E.normalized_field((self.measured_E.max, self.measured_E.min)), H.normalized_field((self.measured_H.max, self.measured_H.min))

        return {"E": abs(((targ_E-rec_E)**2).mean()), "H": abs(((targ_H-rec_H)**2).mean())}
    
    def plot_reconstruction(self, field_components="norm", height=0):
        compare_reconstruction(
            H_target=self.measured_H, E_target=self.measured_E, 
            dh_reconstructed=self.fh, field_components=field_components, height=height)
        return self
    
    def plot_reconstruction_H(self, field_components="norm", height=0):
        compare_H(
            H_target=self.measured_H, 
            dh_reconstructed=self.fh, field_components=field_components, height=height)
        return self
    
    def plot_reconstruction_E(self, field_components="norm", height=0):
        compare_E(
            E_target=self.measured_E, 
            dh_reconstructed=self.fh, field_components=field_components, height=height)
        return self
    

import matplotlib.pyplot as plt


def compare_H(
    H_target: Field3D, dh_reconstructed: DFHandler_over_Substrate,
    field_components = "norm", scale="linear", height=0
    ):
    comp = field_components

    # H comparison
    fig, ax = plt.subplots(1,2, figsize=(16,6), constrained_layout=True)
    dh_reconstructed.H.get_2D_field("z", height).contour_plot_magnitude(comp, scale=scale , ax=ax[0])
    H_target.get_2D_field("z", height).contour_plot_magnitude(comp, scale = scale, ax=ax[1])
    fig.suptitle("H field magnitude comparison", weight="bold", size=16)


def compare_E(
    E_target: Field3D, dh_reconstructed: DFHandler_over_Substrate,
    field_components = "norm", scale="linear", height=0
    ):
    comp = field_components

    # E comparison
    fig, ax = plt.subplots(1,2, figsize=(16,6), constrained_layout=True)
    dh_reconstructed.E.get_2D_field("z", height).contour_plot_magnitude(comp, scale=scale , ax=ax[0])
    E_target.get_2D_field("z", height).contour_plot_magnitude(comp, scale = scale, ax=ax[1])
    fig.suptitle("E field magnitude comparison", weight="bold", size=16)

def compare_reconstruction(
    H_target: Field3D, E_target: Field3D, dh_reconstructed: DFHandler_over_Substrate,
    field_components = "norm", scale="linear", height=0
    ):
    comp = field_components


    # H comparison
    fig, ax = plt.subplots(1,2, figsize=(16,6), constrained_layout=True)
    dh_reconstructed.H.get_2D_field("z", height).contour_plot_magnitude(comp, scale=scale , ax=ax[0])
    H_target.get_2D_field("z", height).contour_plot_magnitude(comp, scale = scale, ax=ax[1])
    fig.suptitle("H field magnitude comparison", weight="bold", size=16)

    # E comparison
    fig, ax = plt.subplots(1,2, figsize=(16,6), constrained_layout=True)
    dh_reconstructed.E.get_2D_field("z", height).contour_plot_magnitude(comp, scale=scale , ax=ax[0])
    E_target.get_2D_field("z", height).contour_plot_magnitude(comp, scale = scale, ax=ax[1])
    fig.suptitle("E field magnitude comparison", weight="bold", size=16)