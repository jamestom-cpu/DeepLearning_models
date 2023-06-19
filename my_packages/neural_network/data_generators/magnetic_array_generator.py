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



class RandomMagneticDipoleGenerator(Generator):
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

    def generate_mask(self, n_layers=2, p=None):
        if p is None:
            p = self.dipole_density/n_layers
        mask_layers = [np.random.choice([0, 1], size=(self.resolution), p=[1-p, p]) for _ in range(n_layers)]
        self.mask = np.stack(mask_layers, axis=0)
        self.N_dipoles = np.sum(self.mask)
        return self.mask

    def _generate_random_moments(self):
        moments_abs = np.random.uniform(1/self.dynamic_range, 1, size=(self.N_dipoles,))
        moments_phase = np.random.uniform(0, 2*np.pi, size=(self.N_dipoles,))
        return moments_abs * np.exp(1j * moments_phase)
    
    # def _generate_random_moments(self):
    #     moments_r, moments_i = np.random.uniform(1/self.dynamic_range, 1, size=(2, self.N_dipoles))
    #     return moments_r+1j*moments_i
    
    def _return_r0(self):
        """
        mask: 2D array of 0's and 1's
        grid: Grid object
        """
        x,y,z = self.r0_grid
        
        # find positions of x oriented dipoles
        xmask = self.mask[0]
        x_orientation_r0x = x[xmask == 1]
        x_orientation_r0y = y[xmask == 1]

        self.r0x = np.hstack((x_orientation_r0x, x_orientation_r0y))
        
        # find positions of y oriented dipoles
        ymask = self.mask[1]
        y_orientation_r0x = x[ymask == 1]
        y_orientation_r0y = y[ymask == 1]
        
        self.r0y = np.hstack((y_orientation_r0x, y_orientation_r0y))

        # create a list of x and y oriented dipoles
        x = [x_orientation_r0x.v[:,0], y_orientation_r0x.v[:,0]]
        y = [x_orientation_r0y.v[:,0], y_orientation_r0y.v[:,0]]
        
        x = np.concatenate(x)
        y = np.concatenate(y)
        
        z = np.full_like(x, self.dipole_height)
        self.r0 = np.stack((x, y, z), axis=-1)
        return self.r0
    
    def _return_orientation(self):
        # the mask structure contains the information on the orientation
        xmask = self.mask[0]
        ymask = self.mask[1]
        n_xoriented_dipoles = np.sum(xmask)
        n_yoriented_dipoles = np.sum(ymask)
        xorientations = np.asarray([[np.pi/2, 0]]*int(n_xoriented_dipoles)) 
        yorientations = np.asarray([[np.pi/2, np.pi/2]]*int(n_yoriented_dipoles))

        if xorientations.shape[0] == 0 and yorientations.shape[0] == 0:
            self.orientations = np.asarray([])
        elif xorientations.shape[0] == 0:
            self.orientations = yorientations
        elif yorientations.shape[0] == 0:
            self.orientations = xorientations
        else:
            self.orientations = np.concatenate((xorientations, yorientations))
        return self.orientations


    def _return_magnetic_dipole_array(self, scale_factor=1):
        r0 = self._return_r0()
        orientations = self._return_orientation()
        M = self._generate_random_moments()*scale_factor

        self.magnetic_array = FlatDipoleArray(
            f=self.f, 
            height=self.dipole_height, 
            r0=r0, 
            orientations=orientations, 
            moments=np.expand_dims(M, axis=-1), 
            type="Magnetic"
            )
        
        return self.magnetic_array
    
    
    
    def _generate_dfh(self):
        dipole_array = self._return_magnetic_dipole_array()

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
        self.dfh = self.dfh_full.dh_magnetic
        return self.dfh
    
    def generate_random_H_fields(self):
        self.generate_mask()
        dfh = self._generate_dfh()

        if self.N_dipoles == 0:
            zero_scan = lambda component: Scan(np.zeros(self.field_res), grid=self.r, 
                                               freq=self.f, axis="z", component=component, field_type="H")
            return zero_scan("x"), zero_scan("y")
        H = dfh.evaluate_H()
        Hx = H.run_scan(component= "x", index = self.probe_height, field_type="H")
        Hy = H.run_scan(component= "y", index = self.probe_height, field_type="H")
        return Hx, Hy
    
    def create_a_copy(self):
        return RandomMagneticDipoleGenerator(**self._my_basis_dict)
    
    def generate_labeled_data(self):
        Hx, Hy = self.generate_random_H_fields()
        fields = np.stack((Hx.scan, Hy.scan), axis=0)
        return fields, self.mask
    
    def plot_labeled_data(self, fields, mask, ax=None, FIGSIZE=(10,3), image_folder="images", savename="random_dipole_field.png"):

        Hx = Scan(fields[0], grid=self.r, freq=self.f, axis="z", component="x", field_type="H")
        Hy = Scan(fields[1], grid=self.r, freq=self.f, axis="z", component="y", field_type="H")

        x, y = self.r0_grid[:-1, ..., 0]
        xr0x = x.v[mask[0] == 1]
        xr0y = y.v[mask[0] == 1]

        yr0x = x.v[mask[1] == 1]
        yr0y = y.v[mask[1] == 1]

        if ax is None:
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=FIGSIZE, constrained_layout=True)
        else:
            ax1, ax2 = ax
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        
        Hx.plot(ax=ax1, title="Hx")
        ax1.scatter(xr0x, xr0y, c="r", marker=">", edgecolor="k", label="x oriented")
        ax1.scatter(yr0x, yr0y, c="k", marker="^", label="y oriented")
        
        
        Hy.plot(ax=ax2, title="Hy")
        ax2.scatter(xr0x, xr0y, c="r", marker=">", edgecolor="k", label="x oriented")
        ax2.scatter(yr0x, yr0y, c="k", marker="^", label="y oriented")
        legend = ax2.legend(loc="upper center", bbox_to_anchor=(-0.5, 0.95), frameon=True,
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

    rdg = RandomMagneticDipoleGenerator(
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
    rdg.plot_labeled_data(field, target)

    if PLOT:
        Hx, Hy = rdg.generate_random_H_fields()
        FIGSIZE = (9, 13)

        fig, ax = plt.subplots(5,2, figsize=FIGSIZE, constrained_layout=True)
        image_folder = "images"
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        for ii in range(10):
            Hx, _ = rdg.generate_random_H_fields()
            r0 = rdg._return_r0()
            Hx.plot(ax=ax.flatten()[ii])
            ax.flatten()[ii].scatter(rdg.r0x[:, 0], rdg.r0x[:, 1], c="r", marker=">", edgecolor="k", label="x oriented dipoles")
            ax.flatten()[ii].scatter(rdg.r0y[:, 0], rdg.r0y[:, 1], c="k", marker="^", label="y oriented dipoles")
        ax.flatten()[1].legend(loc="upper left", bbox_to_anchor=(0.5, 1.7))
        fig.suptitle("Hx - probe height: {}".format(probe_height), y=0.98)

        fullpath = os.path.join(image_folder, "Hx_random_dipoles.png")
        plt.savefig(fullpath, dpi=300, bbox_inches="tight")

        fig, ax = plt.subplots(5,2, figsize=FIGSIZE, constrained_layout=True)
        for ii in range(10):
            _, Hy = rdg.generate_random_H_fields()
            r0 = rdg._return_r0()
            Hy.plot(ax=ax.flatten()[ii])
            ax.flatten()[ii].scatter(rdg.r0x[:,0], rdg.r0x[:,1], c="r", marker=">", edgecolor="k", label="x oriented dipoles")
            ax.flatten()[ii].scatter(rdg.r0y[:,0], rdg.r0y[:,1], c="k", marker="^", label="y oriented dipoles")
        ax.flatten()[1].legend(loc="upper left", bbox_to_anchor=(0.5, 1.7))
        fig.suptitle("Hy - probe height: {}".format(probe_height), y=0.98)

        
        fullpath = os.path.join(image_folder, "Hy_random_dipoles.png")
        plt.savefig(fullpath, dpi=300, bbox_inches="tight")
        plt.show()