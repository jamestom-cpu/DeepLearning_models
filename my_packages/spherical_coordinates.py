import numpy as np

def unit_direction_from_theta_phi(orientations: np.array)-> np.ndarray:
    theta, phi = np.array(orientations).T
    r = np.ones_like(theta)

    sph_coords = np.stack([r, theta, phi], axis=0)
    return to_cartesian_grid(sph_coords).T

def to_spherical_grid(grid):
    x,y,z = grid
    R = np.linalg.norm(grid, axis=0)
    elevation = np.arctan2(np.linalg.norm(grid[:-1], axis=0), z)
    azimuth = np.arctan2(y, x)

    spherical_grid = np.stack([R, elevation, azimuth], axis=0)
    return spherical_grid

def to_cartesian_grid(spher_grid):
    r, theta, phi = spher_grid
    x = r * np.sin( theta ) * np.cos( phi )
    y = r * np.sin( theta ) * np.sin( phi )
    z = r * np.cos( theta )

    cartesian_grid = np.stack([x,y,z], axis=0)
    return cartesian_grid

def field_to_cartesian_matrix(grid_sph):

    """
    the transformation matrix is different in each point of space.
    this function calculates the transformation matrix for each point of the grid!
    the final shape of the output is therefore (3,3,grid_shape)
    """

    r, theta, phi = grid_sph
    

    column1 = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
    column1 = np.stack(column1, axis=0)

    column2 = [np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)]
    column2 = np.stack(column2, axis=0)

    column3 = [-np.sin(phi), np.cos(phi), np.zeros(r.shape)]
    column3 = np.stack(column3, axis=0)

    sph2cart = np.stack([column1, column2, column3], axis=0)

    return sph2cart

def vector_field_2cartesian(field_sph, grid_sph):
    """
    the 3D vector field has shape (3, grid_shape, frequencies). In order to transform this into the cartesian coordinate we must multiply each point
    by a transformation matrix T. The transformation matrix T depends on r, theta, phi. 
    Therefore the tensor that contains all the transformation matrices has shape (3,3,grid_shape, frequencies). Finally, for each point in space the 
    we must apply the matrix multiplication. 

    The function that defines the transformation matrix the shape as (LINES, COLUMNS, GRID_SHAPE). Therefore the matrix multiplication
    between field and matrix takes place on the outermost dimension: LINES*COLUMNS_VECTOR_FIELD.

    Therefore the einstein summation can be written as:
    cartesian_field = np.einsum("i..., i...", transf_matrix, field_spherical_coordinates)
    """
    sph2cart_matrix = field_to_cartesian_matrix(grid_sph)

    # transform
    cartesian_field = np.einsum("j...,j...i", sph2cart_matrix, field_sph)
    return cartesian_field
    
