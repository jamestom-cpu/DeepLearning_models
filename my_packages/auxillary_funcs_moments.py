import numpy as np


from my_packages.classes.aux_classes import Grid


from typing import Tuple
from math import gcd
import decimal


def find_decimal_order(val: float):
    return -int(decimal.Decimal(str(val)).as_tuple().exponent)

def return_regular_axis(axis, min_bound=None, max_bound=None)-> Tuple[np.ndarray, np.ndarray]:
    if min_bound is None:
        min_bound = np.min(axis)
    if max_bound is None:
        max_bound = np.max(axis)


    axis = np.unique(axis)

    rounding_digits = 5
    steps = np.diff(axis).round(rounding_digits)
    steps = np.unique(steps)

    if len(axis)==1:
        if max_bound == min_bound:
            max_bound += 1
            min_bound -= 1
            
        steps = np.array([(max_bound-min_bound)/20])
         
            

    k = max([find_decimal_order(step) for step in steps ])
    int_steps = steps*10**k


    gcd_val = int(int_steps[0])
    for i in range(1, len(int_steps)):
        gcd_val = gcd(gcd_val, int(int_steps[i]))
    min_step = gcd_val/10**k


    n_steps = int(np.round((max_bound-min_bound)/min_step))
    diff = (max_bound-min_bound)-n_steps*min_step
    min_bound -= diff/2
    max_bound += diff/2



    regular_axis = np.linspace(min_bound, max_bound, n_steps+1)

    indices = np.zeros(len(axis), dtype=int)
    for ii in range(len(axis)):
        indices[ii] = np.argmin(np.abs(regular_axis-axis[ii]))
    return regular_axis, indices




# def unique_with_tolerance(arr: np.array, atol: float = 1e-5):
#     """Returns the unique elements of an array with a tolerance"""
#     differences = np.diff(arr)
#     return set(arr[np.concatenate(([True], np.abs(differences) > atol))])

# def expand_moments_and_grid(M: np.array, grid: Grid, new_bounds: Grid):
#     steps_x = unique_with_tolerance(np.diff(grid.x), atol=1e-5)
#     steps_y = unique_with_tolerance(np.diff(grid.y), atol=1e-5)

#     z = grid.z

#     assert len(steps_x) == 1, "the grid is not uniform in x: {}".format(steps_x)
#     assert len(steps_y) == 1, "the grid is not uniform in y: {}".format(steps_y)

#     step_x = list(steps_x)[0]
#     step_y = list(steps_y)[0]

#     del steps_x, steps_y

#     (xmin_new, xmax_new), (ymin_new, ymax_new) = new_bounds[:-1]
#     (xmin_old, xmax_old), (ymin_old, ymax_old) = grid.bounds()[:-1]

#     assert xmin_new <= xmin_old, "the new grid is smaller than the old one"
#     assert xmax_new >= xmax_old, "the new grid is smaller than the old one"
#     assert ymin_new <= ymin_old, "the new grid is smaller than the old one"
#     assert ymax_new >= ymax_old, "the new grid is smaller than the old one"

#     Nx0 = int(abs(xmin_old - xmin_new)/step_x)
#     Nx1 = int(abs(xmax_old - xmax_new)/step_x)
#     Ny0 = int(abs(ymin_old - ymin_new)/step_y)
#     Ny1 = int(abs(ymax_old - ymax_new)/step_y)

#     x1_new = np.linspace(xmin_old - Nx0*step_x, xmin_old, Nx0)
#     x2_new = np.linspace(xmax_old, xmax_old + Nx1*step_x, Nx1)
#     xnew = np.concatenate((x1_new, grid.x, x2_new))

#     y1_new = np.linspace(ymin_old - Ny0*step_y, ymin_old, Ny0)
#     y2_new = np.linspace(ymax_old, ymax_old + Ny1*step_y, Ny1)
#     ynew = np.concatenate((y1_new, grid.y, y2_new))

#     new_grid = np.asarray(np.meshgrid(xnew, ynew, z, indexing="ij"))

#     print()


#     Mnew = np.concatenate([np.zeros((len(x1_new), M.shape[1]), dtype=float), M, np.zeros((len(x2_new), M.shape[1]), dtype=float)], axis=0)
#     Mnew = np.concatenate([np.zeros((Mnew.shape[0], len(y1_new)), dtype=float), Mnew, np.zeros((Mnew.shape[0], len(y2_new)), dtype=float)], axis=1)

#     return Mnew, new_grid


from typing import Callable

def custom_conv2d(image, kernel, func_on_roi: Callable):
    """
    Performs 2D convolution on an image using a kernel.

    Args:
    image (numpy.ndarray): Input image of shape (height, width).
    kernel (numpy.ndarray): Convolution kernel of shape (kernel_height, kernel_width).

    Returns:
    numpy.ndarray: Convolved image of shape (height, width).
    """

    # Get dimensions of the image and the kernel
    height, width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate padding for the image to ensure output has the same size
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Create padded image with zeros
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=np.nan)

    # Initialize output image with zeros
    output = np.zeros((height, width))

    # Perform convolution
    for i in range(height):
        for j in range(width):
            # Extract the region of interest (ROI) from the padded image
            roi = padded_image[i:i + kernel_height, j:j + kernel_width]
            # normalize the ROI
            # if np.std(roi) > 1e-1:
            #     roi = (roi - np.mean(roi)) / np.std(roi)
            roi_on_kernel = roi[kernel != 0]
            output[i, j] = func_on_roi(roi_on_kernel)
                

    return output


def min_custom_conv2d(M0, kernel, func_on_roi: Callable):
    conv1 = custom_conv2d(M0, kernel, func_on_roi)
    conv2 = custom_conv2d(M0, kernel.T, func_on_roi)

    assert conv1.shape == conv2.shape, "conv1 and conv2 must have the same shape"
    new_elements = np.zeros(conv1.shape)
    for ii, (elem1, elem2) in enumerate(zip(conv1.flatten(), conv2.flatten())):
        if np.isnan(elem1) and np.isnan(elem2):
            new_elements.flat[ii] = np.nan
        elif np.isnan(elem1):
            new_elements.flat[ii] = elem2
        elif np.isnan(elem2):
            new_elements.flat[ii] = elem1
        else:
            new_elements.flat[ii] = np.min([elem1, elem2])   
    return new_elements.reshape(conv1.shape)


def return_moments_on_grid(M: np.array, r0: np.array):
    assert np.ndim(M) == 1, "M must be a 1D array"
    assert np.ndim(r0) == 2, "r0 must be a 2D array"
    assert len(M) == len(r0), "M and r0 must have the same length"

    # convert to numpy arrays if not already
    M = np.asarray(M)
    r0 = np.asarray(r0)

    x, y, z = r0.T
    
    x_axis = np.unique(x)
    y_axis = np.unique(y)
    z_axis = np.unique(z)

    M_vals = np.zeros((len(x_axis), len(y_axis)), dtype=complex)
    M_vals[:] = np.nan
    grid = np.asarray(np.meshgrid(x_axis, y_axis, z_axis, indexing="ij"))
    for ii, m in enumerate(M):
        x,y,z = r0[ii]
        M_vals[np.where(np.all(np.isclose(grid[:-1, ..., 0], np.array([x,y])[:, None, None]), axis=0))] = m
    return M_vals, Grid(grid)

def return_moments_on_expanded_grid(M, r0, r: Grid=None):
    bounds_x = r.bounds()[0] if r is not None else (None, None)
    bounds_y = r.bounds()[1] if r is not None else (None, None)

    ax_x, indx = return_regular_axis(r0.x, *bounds_x)
    ax_y, indy = return_regular_axis(r0.y, *bounds_y)
    ax_z = r0.z

    expanded_grid = np.asarray(np.meshgrid(ax_x, ax_y, ax_z, indexing="ij"))
    indices_grid = np.asarray(np.meshgrid(indx, indy, indexing="ij"))

    expanded_M = np.zeros((len(ax_x), len(ax_y)), dtype=complex)
    expanded_M[:] = np.nan

    list_of_indices = list(zip(indices_grid[0].flatten(), indices_grid[1].flatten()))
    list_of_indices = np.asarray(list_of_indices).T
    expanded_M[list_of_indices[0], list_of_indices[1]] = M.flatten()

    return expanded_M, Grid(expanded_grid)



def circular_diff(phases, angles_in_degrees=False):
    """Calculate the circular difference of a set of phases.

    Parameters
    ----------
    phases : array_like

    angles_in_degrees : bool, optional
        If True, the input phases are assumed to be in degrees and will be
        converted to radians. Default is False.

    Returns
    -------
    circ_diff : float
        The circular difference.

    Notes
    -----
    The circular difference is a measure of the difference between the phases.
    It is defined as the angle between the mean complex value of the phases
    and the first phase. The circular difference is in the same units as the
    input phases.

    References
    ----------
    .. [1] Fisher, N.I. (1993), Statistical Analysis of Circular Data, Cambridge
              University Press, Cambridge, UK.

    """
    
    if angles_in_degrees:
        phases = np.deg2rad(phases)

    complex_values = np.exp(1j * phases)

    diff = np.diff(complex_values)/np.pi
    return np.sum(np.abs(diff))


def circular_std(phases, angles_in_degrees=False):
    """Calculate the circular standard deviation of a set of phases.

    Parameters
    ----------
    phases : array_like

    angles_in_degrees : bool, optional
        If True, the input phases are assumed to be in degrees and will be
        converted to radians. Default is False.

    Returns
    -------
    circ_std : float
        The circular standard deviation.

    Notes  
    -----
    The circular standard deviation is a measure of the dispersion of the
    phases. It is defined as the square root of the circular variance, which
    is 1 minus the absolute value of the mean complex value of the phases.
    The circular standard deviation is in the same units as the input phases.

    References
    ----------
    .. [1] Fisher, N.I. (1993), Statistical Analysis of Circular Data, Cambridge
              University Press, Cambridge, UK.
    
     """
    
    if angles_in_degrees:
        phases = np.deg2rad(phases)

    # Calculate mean_complex
    mean_complex = np.mean(np.exp(1j * phases))

    # Calculate circular variance
    circ_var = 1 - np.abs(mean_complex)

    # Calculate circular standard deviation
    circ_range = np.pi  # Half of the angle range (in radians)
    circ_std = np.sqrt(circ_var * circ_range**2)

    if angles_in_degrees:
        circ_std = np.rad2deg(circ_std)
    return circ_std