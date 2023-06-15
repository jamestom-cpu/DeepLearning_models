import numpy as np


def curl(field: np.ndarray, x, y, z):
    u,v,w = field

    du_dy, du_dz = np.gradient(u, y, z, axis=[1,2])
    dv_dx, dv_dz = np.gradient(v, x, z, axis=[0,2])
    dw_dx, dw_dy = np.gradient(w, x, y, axis=[0,1])

    rot_x = dw_dy - dv_dz
    rot_y = du_dz - dw_dx
    rot_z = dv_dx - du_dy

    return np.stack([rot_x, rot_y, rot_z], axis=0)

