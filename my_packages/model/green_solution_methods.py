import numpy as np

from my_packages.classes import dipole_array, dipole_fields
from my_packages.classes import field_classes

def solve_with_NormalEquations_matrix(G, target_field, regularization_lambda=0):
    def conj_transpose(X):
        return X.T.conjugate()

    # reshape arrays into matrices
    n_frequencies = G.shape[-1]
    G_matrix = G.reshape(G.shape[0],-1, n_frequencies)
    target_field = target_field.reshape(-1, n_frequencies)

    M=[]
    for f_index in range(n_frequencies):
        Gm = G_matrix[..., f_index].T
        targetm = target_field[..., f_index]

        # normalize G
        Gm_norm = np.linalg.norm(Gm)
        Gm = Gm/Gm_norm

        A = conj_transpose(Gm)@Gm + regularization_lambda*np.eye(Gm.shape[-1])
        b = conj_transpose(Gm)@targetm

        M.append(np.linalg.solve(A, b)/Gm_norm)

    M = np.stack(M, axis=-1)

    return M

    

def get_conditioning_dyadic(Gdyadic, axis=2):
    # flatten green solution tensor into a 2d vector
    n_frequencies = Gdyadic.shape[-1]
    target_field_length = np.prod(Gdyadic.shape[:axis])
    vec_array_length = np.prod(Gdyadic.shape[axis:-1])
    Gdyadic_vector = Gdyadic.reshape(vec_array_length, target_field_length, n_frequencies)

    # the frequency dimension is kept as the first dimension
    Gdyadic_vector = np.moveaxis(Gdyadic_vector, -1, 0)
    condition_number = np.linalg.cond(Gdyadic_vector)

    return condition_number

## For green dyadic
def invert_green_dyadic(Gdyadic, axis=2):

    n_frequencies = Gdyadic.shape[-1]
    target_field_length = np.prod(Gdyadic.shape[:axis])
    vec_array_length = np.prod(Gdyadic.shape[axis:-1])
    Gdyadic_vector = Gdyadic.reshape(vec_array_length, target_field_length, n_frequencies)

    # invert the square matrix with f dimension as axis
    Gdyadic_vector = np.moveaxis(Gdyadic_vector, -1, 0)
    Gdyadic_inv = np.linalg.inv(Gdyadic_vector)
    Ginv = np.moveaxis(Gdyadic_inv, 0, -1)

    # reshape to dimensions
    Gdyadicshape = list(Gdyadic.shape)
    Ginv = Ginv.reshape(Gdyadicshape[axis:-1] + Gdyadicshape[:axis] + [n_frequencies])

    return Ginv

def find_dipole_source_array_by_inv_dyadic(target_field: np.ndarray, Gdyadic: np.ndarray, axis=2):
    G_dyadic_inv = invert_green_dyadic(Gdyadic, axis)
    reconstructed_dipoles = np.einsum("ixyzf,ixyz...f->...f", target_field, G_dyadic_inv)
    return reconstructed_dipoles




def get_field_over_PEC_dyadic(moment_array, d_field):
    Gdyadic = d_field.green_solutions.over_PEC_dyadic()
    field = np.einsum("nif, ni...f->...f", moment_array, Gdyadic)
    return field_classes.Field3D(field, d_field.f, d_field.r)



## For Fixed Moments
def check_conditioning_matrix(G):
    # move frequency to the first axis and flatten the rest of the matrix
    G = np.moveaxis(G, -1, 0)
    Gflat = G.reshape(G.shape[0], G.shape[1], -1) # here G.shape[1] is the number of dipoles

    condition_number = np.linalg.cond(Gflat)
    return condition_number




def find_dipole_source_array_by_inv_matrix(G_tensor: np.ndarray, target_field: np.ndarray):
    G = np.moveaxis(G_tensor, -1, 0)
    Gflat = G.reshape(G.shape[0], G.shape[1], -1)
    Ginv = np.linalg.pinv(Gflat)
    Ginv = np.moveaxis(Ginv, 0, -1)

    flat_target_field = target_field.reshape(-1, target_field.shape[-1])
    reconstructed_sources = np.einsum("ijk, ik -> jk", Ginv, flat_target_field)
    return reconstructed_sources

def get_field_over_PEC(moments, orientations, r0, r, f, height=0):
    initial_dipole_array = dipole_array.DipoleSourceArray(f=f, r0=r0, orientations=orientations, moments=moments)
    initial_field = dipole_fields.InfCurrentLoop(r=r, dipole_array=initial_dipole_array)
    return initial_field.evaluate_magnetic_field_over_PEC(height)

def MSEloss(prediction, target):
    return (np.abs(prediction.flatten()-target.flatten())**2).mean()
