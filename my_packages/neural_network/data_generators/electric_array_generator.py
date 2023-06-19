import numpy as np
from typing import Iterable, Tuple
import os, sys

main_workspace_path = "/workspace"
sys.path.append(main_workspace_path)


from my_packages.classes.aux_classes import Grid
from my_packages.classes.dipole_array import FlatDipoleArray
from my_packages.classes.dipole_fields import DFHandler_over_Substrate
from my_packages.classes.model_components import UniformEMSpace, Substrate
from my_packages.classes.field_classes import Scan

from my_packages.neural_network.data_generators.abstract import Generator
import matplotlib.pyplot as plt



class RandomElectricDipoleGenerator(Generator):
    def __init__(
            self, resolution, 
            xbounds, ybounds, 
            dipole_height, 
            substrate_thickness,
            substrate_epsilon_r,
            probe_height,
            dynamic_range, f=[1e9],
            field_res = (50,50),
            dipole_density = 0.5
            ) -> None:
        super().__init__(
            resolution=resolution,
            xbounds=xbounds,
            ybounds=ybounds,
            dipole_height=dipole_height,
            substrate_thickness=substrate_thickness,
            substrate_epsilon_r=substrate_epsilon_r,
            probe_height=probe_height,
            dynamic_range=dynamic_range,
            f=f,
            field_res=field_res,
            dipole_density=dipole_density
            )
        
        self._my_basis_dict = self.__dict__.copy()
        # remove values from _my_basis_dict that are not in kwargs
        del self._my_basis_dict["r"]
        del self._my_basis_dict["r0_grid"]

    def generate_mask(self, n_layers=1, p=None):
        if p is None:
            p = self.dipole_density/n_layers
        mask_layers = [np.random.choice([0, 1], size=(self.resolution), p=[1-p, p]) for _ in range(n_layers)]
        self.mask = np.stack(mask_layers, axis=0)
        self.N_dipoles = np.sum(self.mask)
        return self.mask
    
    def _generate_random_moments(self):
        moments_r, moments_i = np.random.uniform(1/self.dynamic_range, 1, size=(2, self.N_dipoles))
        return moments_r+1j*moments_i
    
    def _return_r0(self):
        """
        mask: 2D array of 0's and 1's
        grid: Grid object
        """
        x,y,z = self.r0_grid
        
        # find positions of x oriented dipoles
        zmask = self.mask[0]
        z_orientation_r0x = x[zmask == 1]
        z_orientation_r0y = y[zmask == 1]

        self.r0z = np.hstack((z_orientation_r0x, z_orientation_r0y))
        
        # create a list of x and y oriented dipoles
        x = z_orientation_r0x.v[:,0]
        y = z_orientation_r0y.v[:,0]     
        z = np.full_like(x, self.dipole_height)

        self.r0 = np.stack((x, y, z), axis=-1)
        return self.r0
    
    def _return_orientation(self):
        # the mask structure contains the information on the orientation
        zmask = self.mask[0]
        n_zoriented_dipoles = np.sum(zmask)
        zorientations = np.asarray([[0, 0]]*int(n_zoriented_dipoles)) 

        if zorientations.shape[0] == 0:
            self.orientations = np.asarray([])
        else:
            self.orientations = zorientations
        return self.orientations


    def _return_electric_dipole_array(self, scale_factor=1):
        r0 = self._return_r0()
        orientations = self._return_orientation()
        M = self._generate_random_moments()*scale_factor

        self.electric_array = FlatDipoleArray(
            f=self.f, 
            height=self.dipole_height, 
            r0=r0, 
            orientations=orientations, 
            moments=np.expand_dims(M, axis=-1), 
            type="Electric"
            )
        
        return self.electric_array
    
    
    
    def _generate_dfh(self):
        dipole_array = self._return_electric_dipole_array()

        substrate = Substrate(
            x_size=self.xbounds[1]-self.xbounds[0],
            y_size=self.ybounds[1]-self.ybounds[0],
            thickness=self.substrate_thickness,
            material_name="FR4_epoxy",
            eps_r=self.substrate_epsilon_r
            )
        em_space = UniformEMSpace(
            r=self.r,
            )
        self.dfh_full = DFHandler_over_Substrate(
            EM_space=em_space,  
            substrate=substrate,
            dipole_array=dipole_array
            )
        self.dfh = self.dfh_full.dh_electric
        return self.dfh
    
    def generate_random_E_fields(self, N=10):
        self.generate_mask()
        dfh = self._generate_dfh()

        if self.N_dipoles == 0:
            zero_scan = lambda component: Scan(np.zeros(self.field_res), grid=self.r, 
                                               freq=self.f, axis="z", component=component, field_type="H")
            return zero_scan("z")
        E = dfh.evaluate_E(N=N)
        Ez = E.run_scan(component= "z", index = self.probe_height, field_type="E")
        return Ez
    
    def create_a_copy(self):
        return RandomElectricDipoleGenerator(**self._my_basis_dict)
    
    def generate_labeled_data(self):
        Ez = self.generate_random_E_fields()
        fields = np.expand_dims(Ez.scan, axis=0)
        return fields, self.mask
    
    def plot_labeled_data(self, fields, mask, ax=None, FIGSIZE=(10,3), image_folder="images", savename="random_dipole_field.png"):

        Ez = Scan(fields[0], grid=self.r, freq=self.f, axis="z", component="z", field_type="E")
        
        x, y = self.r0_grid[:-1, ..., 0]
        zr0x = x.v[mask[0] == 1]
        zr0y = y.v[mask[0] == 1]


        if ax is None:
            fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
        
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        
        Ez.plot(ax=ax, title="Ez")
        ax.scatter(zr0x, zr0y, c="w", marker="s", edgecolor="k", label="z oriented")
        
        legend = ax.legend(loc="upper center", bbox_to_anchor=(-0.2, 0.95), frameon=True,
                    handlelength=2.5, handletextpad=0.5, ncol=1)

        fullpath = os.path.join(image_folder, savename)
        plt.savefig(fullpath, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    plt.switch_backend('TkAgg')

    resolution=(21,21)
    field_res = (50,50)
    xbounds = [-0.01, 0.01]
    ybounds = [-0.01, 0.01]
    dipole_height = 1e-3
    substrate_thickness = 1.4e-2
    substrate_epsilon_r = 4.4
    dynamic_range = 10
    probe_height = 0.3e-2
    dipole_density = 0.2

    rdg = RandomElectricDipoleGenerator(
        resolution=resolution,
        xbounds=xbounds,
        ybounds=ybounds,
        dipole_height=dipole_height,
        substrate_thickness=substrate_thickness,
        substrate_epsilon_r=substrate_epsilon_r,
        dynamic_range=dynamic_range,
        probe_height=probe_height,
        field_res=field_res,
        dipole_density=dipole_density
        )

    PLOT = True

    field, target = rdg.generate_labeled_data()
    rdg.plot_labeled_data(field, target, savename="random_electric_dipoles.png")

    if PLOT:
        Ez = rdg.generate_random_E_fields()
        FIGSIZE = (9, 13)

        fig, ax = plt.subplots(5,2, figsize=FIGSIZE, constrained_layout=True)
        image_folder = "images"
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        for ii in range(10):
            Ez = rdg.generate_random_E_fields()
            r0 = rdg._return_r0()
            Ez.plot(ax=ax.flatten()[ii])
            ax.flatten()[ii].scatter(rdg.r0z[:, 0], rdg.r0z[:, 1], c="w", marker="s", edgecolor="k", label="z oriented dipoles")
        ax.flatten()[1].legend(loc="upper left", bbox_to_anchor=(0.5, 1.4))
        fig.suptitle("Ez - probe height: {}".format(probe_height), y=0.98)

        fullpath = os.path.join(image_folder, "Ez_random_dipoles.png")
        plt.savefig(fullpath, dpi=300, bbox_inches="tight")

        plt.show()