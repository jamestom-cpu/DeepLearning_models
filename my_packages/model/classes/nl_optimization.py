from dataclasses import dataclass, field
from typing import Callable, Tuple
from scipy.optimize import least_squares
import numpy as np


from my_packages.auxillary_funcs_moments import min_custom_conv2d, return_moments_on_grid, return_moments_on_expanded_grid, circular_std

@dataclass
class NonLinearOptimization:
    """NonLinearOptimization class"""
    M0: np.ndarray
    r0: np.ndarray
    G: np.ndarray
    target: np.ndarray
    component: str
    prm: np.ndarray = None
    fixed_zero_phase_index: int = None
    prm_opt: np.ndarray = None
    opt_sources: np.ndarray = None
    alpha_lreg_M: float = 0
    alpha_lreg_P: float = 0
    alpha_greg_M: float = 0
    func_on_roi_M: Callable = np.std
    func_on_roi_P: Callable = np.std
    ref_percentage_error: float = 0.05
    G_normalization_factor: float = -1
    field_normalization_factor: float = -1
    sum_phase_reg_terms: bool = False
    history = []
    local_regularization_kernel: np.ndarray = field(default_factory=lambda: np.array([[0, 1, 0], [0,1,0], [0, 1, 0]]))
    iteration_counter: int = 0
    
    def __post_init__(self):
        if not isinstance(self.M0, np.ndarray):
            self.M0 = np.asarray(self.M0, dtype=complex) 
        self.M0 = np.abs(self.M0)
        self.chose_zero_phase_index(overwrite=False)

        if self.prm is None:
            self.prm = self.get_initial_params()


        G_matrix = self.G.reshape(self.G.shape[0], -1)
        u, s, vh = np.linalg.svd(G_matrix, full_matrices=False)
        self.svd = (u, s, vh)

    @property
    def singular_values(self):
        return self.svd[1]
    
    @property
    def s0(self):
        return self.singular_values[0]
    
    def update_reg_parameters(self, **regparams):
        for key, value in regparams.items():
            setattr(self, key, value)
        return self
    
        
    def regularize_G_conditioning(self, reg_coeff=0, G=None, verbose=True):
        if reg_coeff == 0:
            return self
        if G is None:
            G = self.G
        Gshape = G.shape
        
        """Regularize G using truncated SVD"""
        U, s, Vh = self.svd
        # remove singular values below threshold
        # Regularize the singular values
        reg_coeff = reg_coeff * self.s0
        s_reg = (s**2 + reg_coeff**2)/s
        
        if verbose:
            print(f"New condition number: {s_reg[0]/s_reg[-1]:.2f} vs old {s[0]/s[-1]:.2f}")
        G_reg = np.einsum("ij,j,jk->ik", U, s_reg, Vh).reshape(Gshape)
        self.G = G_reg
        return self

    def get_initial_params(self) -> np.ndarray:
        if np.ndim(self.M0)==2:
            self.M0 = self.M0[:, 0]

        phases = np.zeros_like(self.M0, dtype="float")
        # one of the phases can be fixed, so we return all phases except the last one
        return np.concatenate((self.M0, phases[:-1]))
        
        
    def chose_zero_phase_index(self, overwrite=False)-> int:
        if self.fixed_zero_phase_index is None or overwrite:
            self.fixed_zero_phase_index = np.argmax(self.M0)
        return self.fixed_zero_phase_index
    
    # def from_params_to_moments_and_phases(self) -> tuple:
    #     recover M and phase from prm
    #     prm = np.concatenate((self.prm, np.zeros(1)))
    #     M, phase = np.split(prm, 2)
    #     fixed_zero_phase_index = self.chose_zero_phase_index()
    #     phase = np.insert(phase[:-1] , fixed_zero_phase_index, 0)
    #     return M, phase
    

    def calculate_field_magnitude_from_dipole_sources(self, M: np.ndarray, phase: np.ndarray, component: str=None)-> np.ndarray:   
        if component is None:
            component = self.component
        
        # append a zero to M
        component_index = ["xyz".index(c) for c in component]
        M = M * np.exp(1j * phase)
        F = np.abs(np.einsum("n...,n",self.G,M))[component_index,..., 0]
        return F
    
    def normalize_params(self, phase_ref_index=None)->  None:
        assert self.prm is not None, "prm is not set"

        if phase_ref_index is None:
            phase_ref_index = self.chose_zero_phase_index()

        self.M, self.phase = self.from_params_to_moments_and_phases(self.prm, phase_ref_index)
        self.M = self.M / self.field_normalization_factor * self.G_normalization_factor
        self.prm = self.from_moments_and_phases_to_params(self.M, self.phase, phase_ref_index)
    
    def denormalize_params(self, phase_ref_index=None)-> None:
        assert self.prm is not None, "prm is not set"
        if phase_ref_index is None:
            phase_ref_index = self.chose_zero_phase_index()
        self.M, self.phase = self.from_params_to_moments_and_phases(self.prm, phase_ref_index)
        self.M = self.M / self.G_normalization_factor * self.field_normalization_factor
        self.prm = self.from_moments_and_phases_to_params(self.M, self.phase, phase_ref_index)
    
    def normalize_all(self):
        assert self.G_normalization_factor == -1, "G_normalization_factor already set"
        assert self.field_normalization_factor == -1, "field_normalization_factor already set"

        self.G_normalization_factor = self.s0
        self.field_normalization_factor = np.max(self.target)

        self.M0 = self.M0 / self.field_normalization_factor * self.G_normalization_factor
        self.target = self.target / self.field_normalization_factor
        self.G = self.G / self.G_normalization_factor

        if self.prm is not None:
            phase_ref_index = self.fixed_zero_phase_index if self.fixed_zero_phase_index is not None else self.chose_zero_phase_index()
            self.normalize_params(phase_ref_index)

        if self.opt_sources is not None:
            self.opt_sources *= self.G_normalization_factor/self.field_normalization_factor
    
    def undo_normalization(self):
        assert self.G_normalization_factor != -1, "G_normalization_factor not set"
        assert self.field_normalization_factor != -1, "field_normalization_factor not set"

        self.M0 = self.M0 / self.G_normalization_factor * self.field_normalization_factor
        self.target = self.target * self.field_normalization_factor
        self.G = self.G * self.G_normalization_factor

        if self.opt_sources is not None:
            self.opt_sources *= self.field_normalization_factor/self.G_normalization_factor

        if self.prm is not None:
            phase_ref_index = self.fixed_zero_phase_index if self.fixed_zero_phase_index is not None else self.chose_zero_phase_index()
            self.denormalize_params(phase_ref_index)
        
        
        self.G_normalization_factor = -1
        self.field_normalization_factor = -1


    def update_history(self, **kwargs):
        self.history.append(kwargs)
        
    
    def residuals(self, prm: np.ndarray):
        zero_phase_index = self.chose_zero_phase_index(overwrite=False)
        # recover M and phase from prm
        M0, phase = NonLinearOptimization.from_params_to_moments_and_phases(prm, zero_phase_index)

        # complex moments
        M = M0 * np.exp(1j * phase)

        

        F = self.calculate_field_magnitude_from_dipole_sources(M0, phase, self.component)
        square_fitting_residuals = np.abs(F**2 - self.target**2).flatten()


        ## get regularization terms
        
        # reshape M
        M2d, gridM = return_moments_on_grid(M, self.r0)
        M2d, _ = return_moments_on_expanded_grid(M2d, gridM)

        phases2D = np.angle(M2d)
        M02D = np.abs(M2d)
        
        if self.alpha_lreg_M not in (None, 0):
            regularization_terms_M = self.local_regularization_residuals(
                M02D, self.local_regularization_kernel, self.alpha_lreg_M, function_on_roi=self.func_on_roi_M
                )
            regularization_terms_M = [np.sum(regularization_terms_M)]
        else:    
            regularization_terms_M = [] 
        
        if self.alpha_lreg_P not in (None, 0):
            regularization_terms_phase = self.local_regularization_residuals(
                phases2D, self.local_regularization_kernel, self.alpha_lreg_P, function_on_roi=self.func_on_roi_P
                )
            if self.sum_phase_reg_terms:
                regularization_terms_phase = [np.sum(regularization_terms_phase)]
        else:
            regularization_terms_phase = []


        # global regularization
        if self.alpha_greg_M not in (None, 0):
            global_reg_terms_M = self.regularization_term_L2(M0, self.alpha_greg_M)
            regularization_terms_M = np.concatenate([regularization_terms_M, [global_reg_terms_M]])

        residuals = np.concatenate([square_fitting_residuals, regularization_terms_M, regularization_terms_phase])

        self.iteration_counter += 1

        if self.iteration_counter%10 == 0:
            self.update_history(M=M0, phase=phase, residuals=residuals)
            print(residuals)
        
        return residuals
    
    
    
    def run_lm_algorithm(self, alpha_lreg_M=None, alpha_lreg_P=None,  max_nfev: int =None,  verbose: bool = 1, *args, **kwargs)-> Tuple[np.ndarray, np.ndarray]:
        """Run the lm algorithm
        
        Parameters
        ----------
        alpha_M : float, optional
            regularization coefficient for the moments, by default None
        alpha_P : float, optional
            regularization coefficient for the phases, by default None
        max_nfev : int, optional
            maximum number of function evaluations, by default None
        verbose : int, optional
            verbosity level, by default 1
        """


        # update the regularization coefficients
        self.alpha_lreg_M = alpha_lreg_M if alpha_lreg_M is not None else self.alpha_lreg_M
        self.alpha_lreg_P = alpha_lreg_P if alpha_lreg_P is not None else self.alpha_lreg_P

        # normalize
        self.normalize_all()



        # run the lm algorithm
        res = least_squares(self.residuals, self.prm, method="lm", max_nfev=max_nfev, verbose=verbose, *args, **kwargs)
        self.iteration_counter = 0

        # recover index of the zero phase
        zero_phase_index = self.chose_zero_phase_index(overwrite=False)
        M0_opt, ph_opt =  self.from_params_to_moments_and_phases(res.x, zero_phase_index)

        # remove the zero phase
        self.opt_sources = M0_opt*np.exp(1j*ph_opt)
        self.prm = np.concatenate((M0_opt, np.delete(ph_opt, zero_phase_index)))

        # undo the normalization
        print("undoing normalization")
        self.undo_normalization()

        return self.opt_sources
    
    
    @staticmethod
    def from_params_to_moments_and_phases(prm: np.ndarray, zero_phase_index: int = None) -> tuple:
        # recover M and phase from prm
        prm = np.concatenate((prm, np.zeros(1)))
        M, phase = np.split(prm, 2)

        if zero_phase_index is None:
            zero_phase_index = np.argmax(M)

        phase = np.insert(phase[:-1] , zero_phase_index, 0)
        return M, phase
    
    @staticmethod
    def from_moments_and_phases_to_params(M: np.ndarray, phase: np.ndarray, zero_phase_index: int = None) -> np.ndarray:
        if zero_phase_index is None:
            zero_phase_index = np.argmax(M)
        ref_phase = phase[zero_phase_index]
        phase = phase - ref_phase

        # keep phase between -180 and 180
        phase = np.mod(phase + 180, 360) - 180

        phase = np.delete(phase, zero_phase_index)
        prm = np.concatenate((M, phase))
        return prm

    @staticmethod
    def extract_largest_singular_value(matrix):
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        largest_singular_value = s[0]
        return largest_singular_value
    
    @staticmethod
    def regularization_term_L2(x: np.ndarray, alpha: float):
        # ridge regression or L2 regularization
        return alpha*np.sum(np.square(x))
    
    @staticmethod
    def local_regularization_residuals(x: np.ndarray, kernel: np.ndarray, alpha: float, function_on_roi: Callable = np.sum):
        # ridge regression or L2 regularization
        loss_out = min_custom_conv2d(x, kernel, func_on_roi=function_on_roi)
        reg_residuals = loss_out[np.isfinite(loss_out)].flatten()
        return alpha*reg_residuals