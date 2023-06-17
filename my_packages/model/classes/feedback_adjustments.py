
import numpy as np
import matplotlib.pyplot as plt


from my_packages.classes.dipole_array import FlatDipoleArray
from my_packages.classes.dipole_fields import DFHandler_over_Substrate


from my_packages.model.aux_funcs_feedback_adjustments import return_strongest_dipoles, conservative_peak_finder, downsample_error_map
from my_packages.model.classes.nl_optimization import NonLinearOptimization
from my_packages.classes.field_classes import Scan
class SecondStageSearch():
    def __init__(
            self, df_handler: DFHandler_over_Substrate, 
            nl_optimizerE: NonLinearOptimization, 
            nl_optimizerH: NonLinearOptimization, 
            Gcondition_number: float,
            max_nfev: int = 5000,
            intermediate_nfev: int = 500,
            peak_finding_conservativeness: float = 0.75,
            dipole_source_res_shape: tuple = None,
            number_of_clusters_for_simplifcation: int = 4
            ):
        self.max_nfev = max_nfev
        self.dfh = df_handler
        self.nl_optimizerE = nl_optimizerE
        self.nl_optimizerH = nl_optimizerH
        self.Gcondition_number = Gcondition_number
        self.peak_finding_conservativeness = peak_finding_conservativeness
        self.intermediate_nfev = intermediate_nfev
        self.dipole_source_res_shape = dipole_source_res_shape
        self.number_of_clusters_for_simplifcation = number_of_clusters_for_simplifcation

        self.get_target_scans()

    @property
    def Me(self):
        return self.dfh.electric_array.M
    
    @property
    def Mh(self):
        return self.dfh.magnetic_array.M
    
    def _run_lm_electric(self, max_nfev=None, **kwargs):
        if max_nfev is None:
            max_nfev = self.max_nfev
        Me = self.nl_optimizerE.regularize_G_conditioning(
            self.Gcondition_number).run_lm_algorithm(
                max_nfev=max_nfev, **kwargs)
        return Me
    
    def _run_lm_magnetic(self, max_nfev=None, **kwargs):
        if max_nfev is None:
            max_nfev = self.max_nfev
        Mh = self.nl_optimizerH.regularize_G_conditioning(
            self.Gcondition_number).run_lm_algorithm(
                max_nfev=max_nfev, **kwargs)
        return Mh
    
    def optimize_sources(self, max_nfev=None, **kwargs):
        if max_nfev is None:
            max_nfev = self.max_nfev
        Me = self._run_lm_electric(max_nfev=max_nfev, **kwargs)
        Mh = self._run_lm_magnetic(max_nfev=max_nfev, **kwargs)

        electric_darray = self.dfh.electric_array//Me
        magnetic_darray = self.dfh.magnetic_array//Mh
        new_darray = electric_darray + magnetic_darray
        self.dfh = self.dfh.update_dipole_array(new_darray)

    def reduce_complexity(self, **kwargs):
        electric_array = return_strongest_dipoles(self.dfh.electric_array, num_clusters=self.number_of_clusters_for_simplifcation)
        magnetic_array = return_strongest_dipoles(self.dfh.magnetic_array, num_clusters=self.number_of_clusters_for_simplifcation)
        new_array = electric_array + magnetic_array
        self.dfh = self.dfh.update_dipole_array(new_array)
    
    def _return_arguments_for_optimization_electric(self, f_index=0):
        Me = np.asarray(self.dfh.electric_array.M)[..., f_index]
        r0 = self.dfh.electric_array.r0
        field_type = "E"
        components = ["z"]
        G = self.dfh.G_components()["Gee"][...,f_index]
        target = self.nl_optimizerE.target
        return Me, r0, G, field_type, components, target
    
    def _return_arguments_for_optimization_magnetic(self, f_index=0):
        Mh = np.asarray(self.dfh.magnetic_array.M)[..., f_index]
        r0 = self.dfh.magnetic_array.r0
        field_type = "H"
        components = ["x", "y"]
        G = self.dfh.G_components()["Ghh"][...,f_index]
        target = self.nl_optimizerH.target
        return Mh, r0, G, field_type, components, target
    
    @staticmethod
    def _remove_keys_from_dict(dictionary, keys):
        new_dict = dictionary.copy()
        for key in keys:
            if key in new_dict:    
                del new_dict[key]
        return new_dict
    
    @staticmethod
    def _update_nl_optimizer(M, r, G, field_type, component, target, nl_optimizer):
        new_kwargs = {
            "M0": M,
            "r0": r,
            "G": G,
            "component": component,
            "target": target
        } 

        new_nl = NonLinearOptimization(**new_kwargs)
        new_nl.update_reg_parameters(
            alpha_lreg_M = nl_optimizer.alpha_lreg_M,
            alpha_lreg_P = nl_optimizer.alpha_lreg_P,
            alpha_greg_M = nl_optimizer.alpha_greg_M,
            func_on_roi_M = nl_optimizer.func_on_roi_M,
            func_on_roi_P = nl_optimizer.func_on_roi_P
            )
        return new_nl
    
    def update_nl_optimizers(self):
        self.nl_optimizerE = self._update_nl_optimizer(*self._return_arguments_for_optimization_electric(), self.nl_optimizerE)
        self.nl_optimizerH = self._update_nl_optimizer(*self._return_arguments_for_optimization_magnetic(), self.nl_optimizerH)


    def reconstruct_fields(self, f_index=0):
        self.dfh.evaluate_fields()
        probe_height = self.dfh.r.z[0]
        self.Ez_rec = self.dfh.E.run_scan(component="z", index = probe_height)
        self.Hy_rec = self.dfh.H.run_scan(component="y", index = probe_height)
        self.Hx_rec = self.dfh.H.run_scan(component="x", index = probe_height)

    def get_target_scans(self):
        self.Ez_target = Scan(self.nl_optimizerE.target[0], self.dfh.r[..., 0], self.dfh.f, axis="z", component="z", field_type="E")
        self.Hy_target = Scan(self.nl_optimizerH.target[1], self.dfh.r[..., 0], self.dfh.f, axis="z", component="x", field_type="H")
        self.Hx_target = Scan(self.nl_optimizerH.target[0], self.dfh.r[..., 0], self.dfh.f, axis="z", component="y", field_type="H")

    def plot_reconstructed_fields_vs_target(self, f_index=0):
        fig, ax = plt.subplots(2,3, figsize=(15,5), constrained_layout=True)
        self.Ez_rec.plot(ax=ax[0,0], title="Ez reconstructed")
        self.Ez_target.plot(ax=ax[1,0], title="Ez target")
        self.Hx_rec.plot(ax=ax[0,1], title="Hx reconstructed")
        self.Hx_target.plot(ax=ax[1,1], title="Hx target")
        self.Hy_rec.plot(ax=ax[0,2], title="Hy reconstructed")
        self.Hy_target.plot(ax=ax[1,2], title="Hy target")
        fig.suptitle(f"Reconstruction vs target at f={self.dfh.f[f_index]}")

    def calculate_error(self, f_index=0):
        Hmax = np.max([np.max(self.Hx_target.scan), np.max(self.Hy_target.scan)])
        self.Ez_error = (self.Ez_rec - self.Ez_target)/np.max(self.Ez_target.scan)
        self.Hx_error = (self.Hx_rec - self.Hx_target)/Hmax
        self.Hy_error = (self.Hy_rec - self.Hy_target)/Hmax
    

    def downsample_error_map(self, target_shape: tuple):
        # downsample the error to have the same resolution as the dipole array resolution
        Ez_error_vals = downsample_error_map(self.Ez_error, target_shape)
        Hx_error_vals = downsample_error_map(self.Hx_error, target_shape)
        Hy_error_vals = downsample_error_map(self.Hy_error, target_shape)

        self.Ez_error = self.Ez_error.resample_on_shape(target_shape).update_with_scan(Ez_error_vals)
        self.Hx_error = self.Hx_error.resample_on_shape(target_shape).update_with_scan(Hx_error_vals)
        self.Hy_error = self.Hy_error.resample_on_shape(target_shape).update_with_scan(Hy_error_vals)

    def find_new_dipole_poisitions(self, conserv=0.75):
        p_indicesEz = conservative_peak_finder(self.Ez_error.scan, conservativeness=conserv)
        p_indicesHx = conservative_peak_finder(self.Hx_error.scan, conservativeness=conserv)
        p_indicesHy = conservative_peak_finder(self.Hy_error.scan, conservativeness=conserv)

        E_points = np.array([list(self.Ez_error.grid[:-1, x, y]) for x, y in p_indicesEz])
        Hx_points = np.array([list(self.Hx_error.grid[:-1, x, y]) for x, y in p_indicesHx])
        Hy_points = np.array([list(self.Hy_error.grid[:-1, x, y]) for x, y in p_indicesHy])


        # remove any points already occupied by dipoles
        Hy_taken = [point for ii, point in enumerate(self.dfh.magnetic_array.r02) if np.allclose(self.dfh.magnetic_array.orientations[ii], [np.pi/2, np.pi/2])]
        Hx_taken = [point for ii, point in enumerate(self.dfh.magnetic_array.r02) if np.allclose(self.dfh.magnetic_array.orientations[ii], [np.pi/2, 0])]


        E_points = [point for point in E_points if not any(np.isclose(point, self.dfh.electric_array.r02).all(axis=1))]
        Hx_points = [point for point in Hx_points if not any(np.isclose(point, Hx_taken).all(axis=1))]
        Hy_points = [point for point in Hy_points if not any(np.isclose(point, Hy_taken).all(axis=1))]


        dipole_height = self.dfh.dipole_array.height
        E_points = np.hstack([E_points, np.full((len(E_points), 1), dipole_height)]) if len(E_points) > 0 else []
        Hx_points = np.hstack([Hx_points, np.full((len(Hx_points), 1), dipole_height)]) if len(Hx_points) > 0 else []
        Hy_points = np.hstack([Hy_points, np.full((len(Hy_points), 1), dipole_height)]) if len(Hy_points) > 0 else []

        self.new_e_dipole_positions = E_points
        self.new_hx_dipole_positions = Hx_points
        self.new_hy_dipole_positions = Hy_points

        return E_points, Hx_points, Hy_points



    def add_dipoles_to_models(self, new_e_dipole_positions=None, new_hx_dipole_positions=None, new_hy_dipole_positions=None):
        if new_e_dipole_positions is None:
            E_points = self.new_e_dipole_positions
        else:
            E_points = new_e_dipole_positions
        if new_hx_dipole_positions is None:
            Hx_points = self.new_hx_dipole_positions
        else:
            Hx_points = new_hx_dipole_positions
        if new_hy_dipole_positions is None:
            Hy_points = self.new_hy_dipole_positions
        else:
            Hy_points = new_hy_dipole_positions
        
        E_dipoles = self.dfh.electric_array
        H_dipoles = self.dfh.magnetic_array 

        new_Me_value  = np.mean(self.dfh.electric_array.M, axis=0)
        new_Mh_value  = np.mean(self.dfh.magnetic_array.M, axis=0)
        
        
        new_darray_E = FlatDipoleArray(
            r0 = E_points,
            height = E_dipoles.height,
            moments = np.array([new_Me_value for _ in range(len(E_points))]),
            orientations = np.array([[0, 0] for _ in range(len(E_points))]),
            f= self.dfh.f,  
            type="Electric")    
        new_darray_Hx = FlatDipoleArray(
            r0 = Hx_points,
            height = H_dipoles.height,
            moments = np.array([new_Mh_value for _ in range(len(Hx_points))]),
            orientations = np.array([[np.pi/2, 0] for _ in range(len(Hx_points))]),
            f= self.dfh.f,
            type="Magnetic")
        new_darray_Hy = FlatDipoleArray(
            r0 = Hy_points,
            height = H_dipoles.height,
            moments = np.array([new_Mh_value for _ in range(len(Hy_points))]),
            orientations = np.array([[np.pi/2, np.pi/2] for _ in range(len(Hy_points))]),
            f= self.dfh.f,
            type="Magnetic")
        
        darray_E = E_dipoles + new_darray_E
        darray_H = H_dipoles + new_darray_Hx + new_darray_Hy
        
        new_darray = darray_E + darray_H
        self.dfh = self.dfh.update_dipole_array(new_darray)
        return self
    
    def plot_error(self, f_index=0):
        fig, ax = plt.subplots(1,3, figsize=(15,5), constrained_layout=True)
        self.Ez_error.plot(ax=ax[0], title="Ez error")
        self.Hx_error.plot(ax=ax[1], title="Hx error")
        self.Hy_error.plot(ax=ax[2], title="Hy error")
        fig.suptitle(f"Error at f={self.dfh.f[f_index]}")



    def find_best_model(self, intermediate_nfev=None, max_nfev=None, conserv=None, plot=False, res_shape=None, **kwargs):
        if max_nfev is None:
            max_nfev = self.max_nfev
        if res_shape is None:
            res_shape = self.dipole_source_res_shape if self.dipole_source_res_shape is not None else self.Ez_target.shape
        if conserv is None:
            conserv = self.peak_finding_conservativeness
        if intermediate_nfev is None:
            intermediate_nfev = self.intermediate_nfev



        print("initial run")
        self.optimize_sources(max_nfev=intermediate_nfev, **kwargs)

        if plot:
            self.reconstruct_fields()
            self.plot_moments()
            self.plot_reconstructed_fields_vs_target()

        print("reducing complexity")
        self.reduce_complexity()
        self.update_nl_optimizers()

        print("second run")
        self.optimize_sources(max_nfev=intermediate_nfev, **kwargs)
        self.reconstruct_fields()
        if plot:
            self.plot_moments()
            self.plot_reconstructed_fields_vs_target()
        
        self.calculate_error()
        self.downsample_error_map(res_shape)
        if plot:
            self.plot_error()
        print("finding new dipoles")
        self.find_new_dipole_poisitions(conserv=conserv)
        self.add_dipoles_to_models()
        self.update_nl_optimizers()

        print("third run")
        self.optimize_sources(max_nfev=max_nfev, **kwargs)
        self.reconstruct_fields()
        if plot:
            self.plot_moments()
            self.plot_reconstructed_fields_vs_target()

    def plot_moments(self):
        fig, ax = plt.subplots(2,2, figsize=(10,5), constrained_layout=True)
        self.dfh.dh_electric.plot_moment_intensity(ax=ax[0,0])
        self.dfh.dh_magnetic.plot_moment_intensity(ax=ax[1,0])
        self.dfh.dh_electric.plot_moment_phase(ax=ax[0,1])
        self.dfh.dh_magnetic.plot_moment_phase(ax=ax[1,1])
        ax[0,0].set_title("Electric dipoles - moments")
        ax[1,0].set_title("Magnetic dipoles - moments")
        ax[0,1].set_title("Electric dipoles - phases")
        ax[1,1].set_title("Magnetic dipoles - phases")
        fig.suptitle("Dipole moments and phases")