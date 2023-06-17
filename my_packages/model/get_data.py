import numpy as np
import matplotlib.pyplot as plt
import copy

from my_packages.classes import field_classes

def rescale(X, minim, maxim):
        """X should already be distributed between 0 and 1"""
        return (X-0.5)*(maxim-minim)+(maxim+minim)/2



def get_random_dipole_position_on_plane(N, boundaries: dict, dipole_height):
    X,Y = np.random.random_sample((2, N))
    

    X = rescale(X, boundaries["X"][0], boundaries["X"][1])
    Y = rescale(Y, boundaries["Y"][0], boundaries["Y"][1])
    Z = np.full(N, dipole_height)

    initial_postions = np.stack([X,Y,Z], axis=1)

    return initial_postions


def get_random_initial_orientations(N):

    theta, phi = np.random.random_sample((2, N))  
    theta = rescale(theta, -np.pi, np.pi)
    phi = rescale(phi, 0, np.pi) 

    return np.stack([theta, phi], axis=1)

def random_init_moments(n_dipoles, n_freqs):
    return np.random.random_sample((n_dipoles, n_freqs))


class FieldDataLoader():
    def __init__(self, field3D: field_classes.Field3D, height: float, grid_shape2D: tuple, n_frequencies: int):
        self.field3D=field3D
        self.height = height
        self.grid_shape = grid_shape2D
        self.n_frequencies = n_frequencies

        height_index = np.argmin(np.abs(height - field3D.grid[-1, 0, 0, :])) 

        self.field2D = self.field3D.get_2D_field(axis="z", index = height_index)
        self.field2D.height = self.height
        self.field2D.height_index = height_index
        
        X,Y,Z = self.field3D.grid
        self.Xmin, self.Ymin, self.Zmin = X.min(), Y.min(), Z.min()
        self.Xmax, self.Ymax, self.Zmax = X.max(), Y.max(), Z.max()

        self.fmin = self.field3D.freqs.min()
        self.fmax = self.field3D.freqs.max()
    
    def add_white_gaussian_noise(self, spread):
        noise_spread = abs(spread)

        noise_r = np.random.normal(0, noise_spread, self.field2D.field.shape)
        noise_i = np.random.normal(0, noise_spread, self.field2D.field.shape)

        noised_field_values = self.field2D.field + noise_r + 1j*noise_i
        noised_field = copy.deepcopy(self.field2D)
        noised_field.field = noised_field_values
        
        new_self = copy.deepcopy(self)
        new_self.field2D = noised_field

        return new_self
        
    def __call__(self):
        new_grid = np.array(np.meshgrid(
            np.linspace(self.Xmin, self.Xmax, self.grid_shape[1]), 
            np.linspace(self.Ymin, self.Ymax, self.grid_shape[2]), 
            np.array(self.height),
            indexing="ij"
            )).squeeze()
        new_freqs = np.linspace(self.fmin, self.fmax, self.n_frequencies)

        field2d = copy.deepcopy(self.field2D)
        field2d.resample_on_grid(new_grid)
        field2d.evaluate_at_frequencies(new_freqs)
        return field2d
        

    

    def rescale(self, X, minim, maxim):
        """X should already be distributed between 0 and 1"""
        return (X-0.5)*(maxim-minim)+(maxim+minim)/2