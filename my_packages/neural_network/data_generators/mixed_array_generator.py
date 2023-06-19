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

from my_packages.neural_network.data_generators.magnetic_array_generator import RandomMagneticDipoleGenerator
from my_packages.neural_network.data_generators.electric_array_generator import RandomElectricDipoleGenerator
from my_packages.neural_network.data_generators.abstract import Generator

import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

class MixedArrayGenerator(Generator):
    def __init__(
            self, resolution, 
            xbounds, ybounds, 
            dipole_height, 
            substrate_thickness,
            substrate_epsilon_r,
            probe_height,
            dynamic_range, f=[1e9],
            field_res = (50,50),
            dipole_density_E = 0.5,
            dipole_density_H = 0.5,
            include_dipole_position_uncertainty=True,
            ) -> None:
        
        kwargs = {
            "resolution": resolution,
            "xbounds": xbounds,
            "ybounds": ybounds,
            "dipole_height": dipole_height,
            "substrate_thickness": substrate_thickness,
            "substrate_epsilon_r": substrate_epsilon_r,
            "probe_height": probe_height,
            "dynamic_range": dynamic_range,
            "f": f,
            "field_res": field_res,
        }
        
        super().__init__(**kwargs)

        del self.dipole_density
        self.dipole_density_E = dipole_density_E
        self.dipole_density_H = dipole_density_H

        self._my_basis_dict = self.__dict__.copy()
        # remove values from _my_basis_dict that are not in kwargs
        del self._my_basis_dict["r"]
        del self._my_basis_dict["r0_grid"]

        Hkwargs = kwargs | {"dipole_density": dipole_density_H}
        Ekwargs = kwargs | {"dipole_density": dipole_density_E}

        self.magnetic_generator = RandomMagneticDipoleGenerator(**Hkwargs)
        self.electric_generator = RandomElectricDipoleGenerator(**Ekwargs)

        self.RESCALE_CONSTANT = self.define_scale_constant()

        self.include_dipole_position_uncertainty = include_dipole_position_uncertainty
    
    @staticmethod
    def define_scale_constant():
        power = np.random.normal(2, 0.3)
        return 5**power

    def _add_noise_to_dipole_array_xy(self, dipole_array: FlatDipoleArray):
        xcell = np.diff(self.xbounds) / self.resolution[0]
        ycell = np.diff(self.ybounds) / self.resolution[1]

        # add a uniform random noise from -xcell/2 to xcell/2
        ndipoles = len(dipole_array.dipoles)
        xnoise = np.random.uniform(-xcell/2, xcell/2, size=(ndipoles,))
        ynoise = np.random.uniform(-ycell/2, ycell/2, size=(ndipoles,))

        noise = np.stack((xnoise, ynoise, np.zeros_like(xnoise)), axis=1)

        new_r0 = dipole_array.r0 + noise
        
        # update array position
        dipole_array.r0 = new_r0
        return dipole_array
    
    def _generate_random_array(self, rescale_electric_moments=True):
        scale_factor_H = 1
        scale_factor_E = self.RESCALE_CONSTANT if rescale_electric_moments else 1
        magnetic_array = self.magnetic_generator._return_magnetic_dipole_array(scale_factor=scale_factor_H)
        electric_array = self.electric_generator._return_electric_dipole_array(scale_factor=scale_factor_E)
        self.dipole_array = magnetic_array + electric_array
        if self.include_dipole_position_uncertainty:
            self.dipole_array = self._add_noise_to_dipole_array_xy(self.dipole_array)
        return self.dipole_array
    
    def _generate_fh(self):
        dipole_array = self._generate_random_array()
        substrate = Substrate(
            x_size=self.xbounds[1]-self.xbounds[0],
            y_size=self.ybounds[1]-self.ybounds[0],
            thickness=self.substrate_thickness,
            material_name="substrate_material",
            eps_r=self.substrate_epsilon_r
            )
        em_space = UniformEMSpace(
            r=self.r,
            )
        self.dfh = DFHandler_over_Substrate(
            EM_space=em_space,  
            substrate=substrate,
            dipole_array=dipole_array
            )
        return self.dfh
    
    def _generate_masks(self, density_E=None, density_H=None):
        if density_E is None:
            density_E = self.dipole_density_E
        if density_H is None:
            density_H = self.dipole_density_H
        self.magnetic_generator.generate_mask(p=density_H)
        self.electric_generator.generate_mask(p=density_E)
        return self
    
    @property
    def mask(self):
        return np.concatenate([self.electric_generator.mask, self.magnetic_generator.mask], axis=0)
    
    @property
    def N_dipoles(self):
        return self.magnetic_generator.N_dipoles + self.electric_generator.N_dipoles

    def generate_random_fields(self):
        self._generate_masks()
        self._generate_fh()
        self.dfh.evaluate_fields(N=10)
        Ez = self.dfh.E.run_scan(component="z", field_type="E")
        Hx = self.dfh.H.run_scan(component="x", field_type="H")
        Hy = self.dfh.H.run_scan(component="y", field_type="H")
        return Ez, Hx, Hy

    def input_data_to_Scan(self, data):
        if len(data)==3:
            Ez = Scan(data[0], grid=self.r, freq=self.f, axis="z", component="z", field_type="E")
            Hx = Scan(data[1], grid=self.r, freq=self.f, axis="z", component="x", field_type="H")
            Hy = Scan(data[2], grid=self.r, freq=self.f, axis="z", component="y", field_type="H")
            return Ez, Hx, Hy
        elif len(data)==2:
            Hx = Scan(data[0], grid=self.r, freq=self.f, axis="z", component="x", field_type="H")
            Hy = Scan(data[1], grid=self.r, freq=self.f, axis="z", component="y", field_type="H")
            return None, Hx, Hy
        elif len(data)==1:
            Ez = Scan(data[0], grid=self.r, freq=self.f, axis="z", component="z", field_type="E")
            return Ez, None, None

    
    def generate_labeled_data(self):
        Ez, Hx, Hy = self.generate_random_fields()
        fields = np.stack((Ez.scan, Hx.scan, Hy.scan), axis=0)
        return fields, self.mask
    
    def plot_labeled_data(self, fields, mask, ax=None, FIGSIZE=(15,3), image_folder="images", savename="temp.png"):
        Ez, Hx, Hy = self.input_data_to_Scan(fields)
        x, y = self.r0_grid[:-1, ..., 0]
        self._plotting_func(Ez, Hx, Hy, x, y, mask, ax=ax, FIGSIZE=FIGSIZE)
        
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        fullpath = os.path.join(image_folder, savename)
        plt.savefig(fullpath, dpi=300, bbox_inches="tight")

    def plot_Hlabeled_data(self, fields, mask, ax=None, FIGSIZE=(15,3), image_folder="images", savename="temp.png"):
        _, Hx, Hy = self.input_data_to_Scan(fields)
        x, y = self.r0_grid[:-1, ..., 0]
        self._H_plotting_func(Hx, Hy, x, y, mask, ax=ax, FIGSIZE=FIGSIZE)

        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        
        fullpath = os.path.join(image_folder, savename)
        plt.savefig(fullpath, dpi=300, bbox_inches="tight")
    
    def plot_Elabeled_data(self, fields, mask, ax=None, FIGSIZE=(15,3), image_folder="images", savename="temp.png"):
        Ez, _, _ = self.input_data_to_Scan(fields)
        x, y = self.r0_grid[:-1, ..., 0]
        self._E_plotting_func(Ez, x, y, mask, ax=ax, FIGSIZE=FIGSIZE)

        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        
        fullpath = os.path.join(image_folder, savename)
        plt.savefig(fullpath, dpi=300, bbox_inches="tight")

    def create_a_copy(self):
        return MixedArrayGenerator(**self._my_basis_dict)


    @staticmethod
    def _E_plotting_func(Ez: Scan, x, y, mask, ax=None, FIGSIZE=(15,3)):
        zr0x = x.v[mask[0] == 1]
        zr0y = y.v[mask[0] == 1]

        if ax is None:
            fig, ax1 = plt.subplots(1,1, figsize=FIGSIZE, constrained_layout=True)
        else:
            ax1 = ax
        
        Ez.plot(ax=ax1, title="Ez")
        ax1.scatter(zr0x, zr0y, c="w", marker="s", edgecolor="k", label="z oriented")

        legend = ax1.legend(loc="upper center", bbox_to_anchor=(-0.5, 0.95), frameon=True,
                    handlelength=2.5, handletextpad=0.5, ncol=1)


    @staticmethod
    def _H_plotting_func(Hx, Hy, x, y, mask, ax=None, FIGSIZE=(15,3)):
        xr0x = x.v[mask[0] == 1]
        xr0y = y.v[mask[0] == 1]

        yr0x = x.v[mask[1] == 1]
        yr0y = y.v[mask[1] == 1]

        if ax is None:
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=FIGSIZE, constrained_layout=True)
        else:
            ax1, ax2 = ax

        Hx.plot(ax=ax1, title="Hx")
        ax1.scatter(xr0x, xr0y, c="r", marker=">", edgecolor="k", label="x oriented")
        ax1.scatter(yr0x, yr0y, c="k", marker="^", label="y oriented")

        Hy.plot(ax=ax2, title="Hy")
        ax2.scatter(xr0x, xr0y, c="r", marker=">", edgecolor="k", label="x oriented")
        ax2.scatter(yr0x, yr0y, c="k", marker="^", label="y oriented")

        legend = ax1.legend(loc="upper center", bbox_to_anchor=(-0.5, 0.95), frameon=True,
                    handlelength=2.5, handletextpad=0.5, ncol=1)
        

    @staticmethod
    def _plotting_func(Ez, Hx, Hy, x, y, mask, ax=None, FIGSIZE=(15,3)):
        zr0x = x.v[mask[0] == 1]
        zr0y = y.v[mask[0] == 1]

        xr0x = x.v[mask[1] == 1]
        xr0y = y.v[mask[1] == 1]

        yr0x = x.v[mask[2] == 1]
        yr0y = y.v[mask[2] == 1]

        if ax is None:
            fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=FIGSIZE, constrained_layout=True)
        else:
            ax1, ax2, ax3 = ax

        Ez.plot(ax=ax1, title="Ez")
        ax1.scatter(zr0x, zr0y, c="w", marker="s", edgecolor="k", label="z oriented")
        ax1.scatter(xr0x, xr0y, c="r", marker=">", edgecolor="k", label="x oriented")
        ax1.scatter(yr0x, yr0y, c="k", marker="^", label="y oriented")
        
        Hx.plot(ax=ax2, title="Hx")
        ax2.scatter(zr0x, zr0y, c="w", marker="s", edgecolor="k", label="z oriented")
        ax2.scatter(xr0x, xr0y, c="r", marker=">", edgecolor="k", label="x oriented")
        ax2.scatter(yr0x, yr0y, c="k", marker="^", label="y oriented")
        
        
        Hy.plot(ax=ax3, title="Hy")
        ax3.scatter(zr0x, zr0y, c="w", marker="s", edgecolor="k", label="z oriented")
        ax3.scatter(xr0x, xr0y, c="r", marker=">", edgecolor="k", label="x oriented")
        ax3.scatter(yr0x, yr0y, c="k", marker="^", label="y oriented")


        legend = ax1.legend(loc="upper center", bbox_to_anchor=(-0.5, 0.95), frameon=True,
                    handlelength=2.5, handletextpad=0.5, ncol=1)
    

if __name__ == "__main__":
    import torch 

    save_dir = "/workspace/NN_data/mixed_array_data"
    fullpath_train = os.path.join(save_dir, "train_and_valid_dataset.pt")
    train_and_valid_dataset = torch.load(fullpath_train)

    from my_packages.neural_network.datasets_and_loaders.dataset_transformers_H import H_Components_Dataset
    from my_packages.neural_network.datasets_and_loaders.dataset_transformers_E import E_Components_Dataset

    Hds = H_Components_Dataset(train_and_valid_dataset)
    Eds = E_Components_Dataset(train_and_valid_dataset)


    # data parameters
    resolution=(7,7)
    field_res = (21,21)
    xbounds = [-0.01, 0.01]
    ybounds = [-0.01, 0.01]
    dipole_height = 1e-3
    substrate_thickness = 1.4e-2
    substrate_epsilon_r = 4.4
    dynamic_range = 10
    probe_height = 0.3e-2
    dipole_density_E = 0.2
    dipole_density_H = 0.2


    rmg = MixedArrayGenerator(
        resolution=resolution,
        xbounds=xbounds,
        ybounds=ybounds,
        dipole_height=dipole_height,
        substrate_thickness=substrate_thickness,
        substrate_epsilon_r=substrate_epsilon_r,
        probe_height=probe_height,
        dynamic_range=dynamic_range,
        f=[1e9],
        field_res=field_res,
        dipole_density_E=dipole_density_E,
        dipole_density_H=dipole_density_H
        )
    

    NN = 2
    samples, labels = Hds.scale_to_01()[:NN]
    fig, ax = plt.subplots(NN, 3, figsize=(16,3), constrained_layout=True)
    for ii, (s, t) in enumerate(zip(samples, labels)):
        rmg.plot_Hlabeled_data(s, t, ax=ax[ii][1:])

    samples, labels = Eds.scale_to_01()[:NN]
    for ii, (s, t) in enumerate(zip(samples, labels)):
        rmg.plot_Elabeled_data(s, t, ax=ax[ii][0]) 

    plt.show()

# if __name__ == "__main__":
#     fields, mask = mixed_generator.generate_labeled_data()
#     print(fields.shape)
#     print(mask.shape)
#     mixed_generator.plot_labeled_data(fields, mask)
#     plt.show()


    # Ez, Hx, Hy = mixed_generator.generate_random_fields()
    # fig, ax = plt.subplots(1,3, figsize=(15,5))
    # Ez.plot(ax=ax[0])
    # Hx.plot(ax=ax[1])
    # Hy.plot(ax=ax[2])
    # fig.show()
    print("finished")

