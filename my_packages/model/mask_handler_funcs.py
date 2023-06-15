from typing import Tuple

import numpy as np


from .classes.dipole_cell_grid import DipoleCellGrid

def define_cluster_mask(
    dcell_grid: DipoleCellGrid, shape: Tuple[int, int], 
    center_index: Tuple[float, float] = "centered", 
    orientation="all", dims=3
    ):
    M,N = dcell_grid.shape
    background_shape = tuple([dims, M, N])
    mask = np.zeros(background_shape)

    m, n = shape

    if isinstance(center_index, str):
        cx = M//2; cy = N//2
    
    else: 
        cx, cy = center_index

    # conditions on the centerpoint
    if isinstance(cx, int) and (cy, int):
        print('centerpoint is index')

    # if the center coordinates are given as floats convert them to the indices    
    elif isinstance(cx, float) or (cy, float):
        xcp, ycp = dcell_grid.center_points
        xcp = xcp[:,0]; ycp = ycp[0,:]

        cx = np.argmin(np.abs(xcp-cx)); cy=np.argmin(np.abs(ycp-cy))

    x1_min = max(0, cx-m//2)
    y1_min = max(0, cy-n//2)

    x1_max = min(M, cx+(m//2)) if m%2==0 else min(M, cx+(m//2+1)) 
    y1_max = min(N, cy+(n//2)) if n%2==0 else min(N, cy+(n//2+1))

    # orientation index: if all select all three orientations
    # if orientation is a list that includes "x", select index 0
    # if orientation is a list that includes "y", select index 1
    # if orientation is a list that includes "z", select index 2

    orientation_indices = []

    if isinstance(orientation, str):
        if orientation == "all":
            orientation_indices = [0,1,2]
        else:
            if "x" in orientation:
                orientation_indices.append(0)
            if "y" in orientation:
                orientation_indices.append(1)
            if "z" in orientation:
                orientation_indices.append(2)
            if orientation_indices == []:
                raise ValueError("orientation must be a list of strings, a string that specifies either in 'x,y,z' or the string 'all'") 
        
    elif isinstance(orientation, list):
        if "x" in orientation:
            orientation_indices.append(0)
        if "y" in orientation:
            orientation_indices.append(1)
        if "z" in orientation:
            orientation_indices.append(2)
        if orientation_indices == []:
            raise ValueError("orientation must be a list of strings, a string that specifies either in 'x,y,z' or the string 'all'") 

    mask[orientation_indices, x1_min:x1_max, y1_min:y1_max] = 1 
    return mask.astype("bool")
