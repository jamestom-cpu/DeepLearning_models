import numpy as np 
from my_packages import spherical_coordinates


def get_amplitude_and_orientation(array_dipoles):
    norm_cmplx = np.sqrt(np.einsum("njk, njk -> nk", array_dipoles, array_dipoles))
    _, theta, phi = spherical_coordinates.to_spherical_grid(np.moveaxis(np.real(array_dipoles), 1,0))

    return norm_cmplx, np.stack([theta, phi], axis=1)



def get_unit_vectors(orientations: np.ndarray):
    ors = np.moveaxis(orientations, 1, 0)
    sph_coords = np.concatenate([np.ones((ors[0][None,...].shape)), ors], axis=0)
    unit_vectors = spherical_coordinates.to_cartesian_grid(sph_coords)
    return np.moveaxis(unit_vectors, 0, 1)