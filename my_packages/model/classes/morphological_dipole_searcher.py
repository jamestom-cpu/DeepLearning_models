from dataclasses import dataclass, field
from typing import Dict, List
from types import SimpleNamespace
from copy import deepcopy
import textwrap
import os

import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, filters
import matplotlib.patches as mpatches
import pickle


from my_packages.classes.scan_analysis import ScanAnalyzer
from my_packages.classes.dipole_fields import DFHandler_over_Substrate
from my_packages.classes.dipole_array import FlatDipoleArray
from my_packages.classes.field_classes import Scan
from my_packages.model.classes.nl_optimization import NonLinearOptimization
from my_packages.model.plotting_funcs_for_morph_model import plot_working_scan, turn_to_animation


@dataclass
class MorphDipoleSearcher:
    scans: List[ScanAnalyzer]
    df_handler: DFHandler_over_Substrate
    probe_height: float
    variations_for_composite_mask: Dict[str, List[any]] = None
    max_number_of_iterations: int = 10
    automatically_find_kernel: bool = True
    method: str ="average"
    clip_on_max_ratio: float = 0.8
    min_amplitude: float = 0.1
    damping: float = 0.2
    current_iteration: int = 0
    set_threshold_object: int = 0
    set_resolution_shape: tuple = (21, 21)
    mask_average_quota: int = 2
    mask_average_quota_threshold: float = 0.5
    set_scanned_component_under_analysis: str = None
    masks: List[Scan] = field(default_factory = list)
    N_dipoles: int = 0
    current_iteration: int = 0
    N_allowed_dampings: int = 2
    max_damping: float = 0.5
    damping_patience: int = 5
    """This class is developed to simply interface with the iterative process to extract optimal dipole positions.
    It is based on the ScanAnalyzer class, which is used to extract a dipole position guess from single scan in one shot and 
    relies on the data handler class DFHandler_over_Substrate to evaluate the fields from dipoles and run the new scans.

    The process of obtaining the composite mask requires to specify the filters, functions etc you want to use. 
    You can specify them in the variations_for_composite_mask dictionary, which is a dictionary of lists of functions.
    If you have already used the simple_scan_analyzer to extract the dipole position guess, you default to the
    variations_for_composite_mask attribute of the simple_scan_analyzer if no variations are specified.
    
    All optimized solutions will be placed in the SimpleNameSpace attribute self.opt.
    """


    def __post_init__(self):
        self.scans = [self.scans] if isinstance(self.scans, ScanAnalyzer) else self.scans
        # sort the scans such that x is always first, then y and then z
        self.scans = sorted(self.scans, key=lambda x: "xyz".index(x.main.component))
        

        # dafualt for the scanned_component_under_analysis is the first component in the scans
        self.scanned_component_under_analysis = self.scanned_components[0] if self.set_scanned_component_under_analysis is None else self.set_scanned_component_under_analysis
        
        self.variations_for_composite_mask = self.variations_for_composite_mask or self.simple_scan_analyzer.variations_for_composite_mask
        if self.simple_scan_analyzer.kernel is None:
            self.simple_scan_analyzer.get_kernel_from_dh(self.df_handler, self.probe_height)
        self.threshold_object = self.set_threshold_object or self.simple_scan_analyzer.threshold_object
        del self.set_threshold_object
        self.opt = SimpleNamespace(dfh=None)
    
    
    
    @property
    def scanned_components(self):
        # all components are either x or y or z
        # order the components such that x is always first, then y and then z
        components = [scan.main.component for scan in self.scans]
        components = sorted(components, key=lambda x: "xyz".index(x))
        return components
    
    @property
    def scanned_component_under_analysis(self):
        return self._scanned_component_under_analysis
    
    @scanned_component_under_analysis.setter
    def scanned_component_under_analysis(self, value):
        if value not in self.scanned_components:
            raise ValueError(f"scanned_component_under_analysis with value {value} must be one of {self.scanned_components}")
        self._scanned_component_under_analysis = value
        self.simple_scan_analyzer = self.scans[self.scanned_components.index(self.scanned_component_under_analysis)]
        if hasattr(self, value):
            self.history = getattr(self, value).history
        

    @property
    def threshold_object(self):
        return self.simple_scan_analyzer.threshold_object
    
    @threshold_object.setter
    def threshold_object(self, value):
        self.simple_scan_analyzer.threshold_object = value
    
    @property
    def resolution_shape(self):
        return self.simple_scan_analyzer.resolution_shape
    
    @resolution_shape.setter
    def resolution_shape(self, value):
        self.simple_scan_analyzer.resolution_shape = value

    def select_scanned_component(self, component: str)-> "MorphDipoleSearcher":
        assert component in self.scanned_components, f"component must be one of {self.scanned_components}"
        self.scanned_component_under_analysis = component
        return self
    
    def get_best_mask(self):
        if self.threshold_object == -1 and self.current_iteration>0:
            Bkernel = self.history[-1]["binarized_kernel"]
        else:
            Bkernel = None
        return self.simple_scan_analyzer.extract_composite_mask(
            self.variations_for_composite_mask, 
            method=self.method, Bkernel=Bkernel, verbose=False
            ).mask
    
    def _check_if_mask(self, mask):
        return self.simple_scan_analyzer._check_if_mask(mask)

    def return_mask_dipoles_dhandler(self, mask: Scan=None)->DFHandler_over_Substrate:
        mask = mask if mask is not None else self.simple_scan_analyzer.mask
        mask = self._check_if_mask(mask)
        r02d, field_values = self.simple_scan_analyzer.return_dipole_position_values(mask,  clip_quota=self.clip_on_max_ratio, min_amplitude=self.min_amplitude)
        
        # get 3d positions of the dipoles
        dipole_array = self.df_handler.dipole_array
        if isinstance(dipole_array, FlatDipoleArray):
            dipole_height = dipole_array.height
        else:
            dipole_height = self.df_handler.dipole_array.dipoles[0].r0[2]
        r03 = np.ones((len(r02d), 1))*dipole_height
        r0 = np.concatenate([r02d, r03], axis=1)


        orientation = self.simple_scan_analyzer.main.component
        field_type = "Electric" if self.simple_scan_analyzer.main.field_type == "E" else "Magnetic"
        dipoles = []
        for R, V in zip(r0, field_values):
            handler = self.df_handler.get_handler_for_single_dipole(orientation, field_type, r0=R)
            dipoles.append((handler.dipole_array*V).dipoles[0])
        
        darray = FlatDipoleArray.init_dipole_array_from_dipole_list(self.df_handler.f, dipoles)
        new_handler = DFHandler_over_Substrate(EM_space=self.df_handler.EM_space, substrate=self.df_handler.substrate, dipole_array=darray)
        return new_handler.evaluate_fields()

    def remove_mask_effect_from_dipole_array(self, scan: Scan = None, mask: Scan = None, damping:float = None)-> Scan:
        mask = mask if mask is not None else self.simple_scan_analyzer.mask
        scan = scan if scan is not None else self.simple_scan_analyzer.main
        damping = damping if damping is not None else self.damping
        mask = self._check_if_mask(mask)
        
        new_fh_handler = self.return_mask_dipoles_dhandler(mask)
        
        mask_field = getattr(
            new_fh_handler, self.simple_scan_analyzer.main.field_type
            ).run_scan(
            axis=self.simple_scan_analyzer.main.axis, 
            component=self.simple_scan_analyzer.main.component, 
            index=self.probe_height
            ).normalize()
        return (self.simple_scan_analyzer.main.normalize() - mask_field*damping).normalize()
    
    def for_every_scanned_component(func):
        def wrapper(self, *args, scan_component_to_consider:str="all", **kwargs):
            if scan_component_to_consider == "all":
                scan_component_to_consider = self.scanned_components
            else:
                for c in scan_component_to_consider:
                    if c not in self.scanned_components:
                        raise ValueError(f"scanned_component {c} must be one of {self.scanned_components}")
            for c in scan_component_to_consider:
                self.scanned_component_under_analysis = c
                self = func(self, *args, **kwargs)
                if len(scan_component_to_consider) >1: 
                    setattr(self, f"{c}", SimpleNamespace())
                    getattr(self, c).history = self.history
                    getattr(self, c).current_iteration = self.current_iteration
                    getattr(self, c).masks = self.masks 
                    getattr(self, c).N_dipoles = self.N_dipoles        
                    self.history = []
            return self
        return wrapper
        
        
    @for_every_scanned_component
    def find_best_dipole_positions(self, resolution_shape=None, number_of_iterations: int = None):
        number_of_iterations = number_of_iterations or self.max_number_of_iterations
        
        if resolution_shape is not None:
            self.resolution_shape = resolution_shape

        original_main_field = self.simple_scan_analyzer.main.scan.copy()
        masks = []
        # cumulative_masks = []
        self.history = []
        working_scan = self.simple_scan_analyzer.main
        N_dampings = 0
        last_damp = 0
        for i in range(number_of_iterations):
            self.current_iteration = i
            composite_mask = self.get_best_mask()
            reset_damping = False

            # if no new dipoles are found in the last 5 iterations, stop the algorithm
            if len(self.history)>(last_damp+self.damping_patience) and all([int(h["num_new_dipoles"]) == 0 for h in self.history[-self.damping_patience:]]):
                print("No new dipoles found in the last 5 iterations, setting damping to {}".format(self.max_damping))
                backup_damping = self.damping
                self.damping = self.max_damping 
                reset_damping = True
                N_dampings += 1
            
            if N_dampings > self.N_allowed_dampings:
                print("No new dipoles found in the last {} iterations, stopping at iteration {}".format(N_dampings, i))
                break

            # run checks on the new composite mask before continuing
            answers = self.run_checks(masks = masks, composite_mask = composite_mask, iteration = i)
            if isinstance(answers, int) and answers == -3:
                new_dip = [h["num_new_dipoles"] for h in self.history]
                if not np.any(np.asarray(new_dip[-3:])==0) and np.sum(new_dip)>3:
                    print("too many new dipoles found in the last 3 iterations, stopping at iteration {}".format(i))

                    # find latest flat point
                    latest_flat_index = np.where(np.asarray(new_dip)==0)[0].max()

                    
                    self.history = self.history[:latest_flat_index]
                    masks = masks[:latest_flat_index]
                    break
                break
            

            if isinstance(answers, int) and answers < 0:
                break
            latest_total_mask, difference_mask, num_new_dipoles = answers

            masks.append(composite_mask)
            #update history
            self.update_history(
                mask = composite_mask, working_scans = working_scan, 
                Bscan =self.simple_scan_analyzer.Bscan, 
                Bkernel = self.simple_scan_analyzer.Bkernel,
                latest_total_mask = latest_total_mask, difference_mask = difference_mask, 
                num_new_dipoles=num_new_dipoles, print_log=True
                )
            
            # if len(self.history)>5 and all([int(h["num_new_dipoles"]) == 0 for h in self.history[-5:]]):
            #     print("No new dipoles found in the last 5 iterations, stopping at iteration {}".format(i))
                
                
                

            
                
                
            #update the scan analyzer
            working_scan = self.remove_mask_effect_from_dipole_array(self.simple_scan_analyzer.main, latest_total_mask)
            self.simple_scan_analyzer.main = working_scan

            
            if reset_damping:
                self.damping = backup_damping
                reset_damping = False
                last_damp = i
                


            
        self.simple_scan_analyzer.main = self.simple_scan_analyzer.main.update_with_scan(original_main_field)

        if masks == []:
            print("No mask found, stopping at iteration {}".format(i))
            self.N_iterations = 0
            self.masks = masks
            self.final_mask = np.zeros_like(self.simple_scan_analyzer.main.scan)
            self.N_dipoles = 0
            self.update_history(
                mask = self.final_mask, working_scans = working_scan, 
                Bscan = self.simple_scan_analyzer.Bscan, 
                Bkernel = self.simple_scan_analyzer.Bkernel,
                latest_total_mask = self.final_mask, difference_mask = None, 
                num_new_dipoles=0, print_log=False
                )
            return self
        

        self.N_iterations = i
        self.masks = masks
        self.final_mask = self.chose_best_mask([h["total_mask"] for h in self.history])
        self.simple_scan_analyzer.mask = self.final_mask
        self.N_dipoles = self.final_mask.sum()
        return self
    

    def run(self, resolution_shape=None, number_of_iterations: int = None, *args, **kwargs):
        self.find_best_dipole_positions(resolution_shape=resolution_shape, number_of_iterations=number_of_iterations)
        self.lm_optimize_source_values(*args, **kwargs)
        for c in self.scanned_components:
            _ = self.to_scan(c) 
        return self

    
    def chose_best_mask(self, masks,  mask_average_quota=None, mask_average_quota_threshold=None):
        """how to select the final mask from the list of masks
        
        Parameters
        ----------
        masks : list of Scan
            list of masks
        mask_average_quota : int, optional
            how many masks to average, by default equal to the class attribute. Set to -1 to disable averaging
            and return boolean_max(mask), that is return a 1 if any of the mask has a 1 in that position
        mask_average_quota_threshold : float, optional
            threshold for the average mask, by default equal to the class attribute. Set to -1 to disable averaging
            and return boolean_max(mask), that is return a 1 if any of the mask has a 1 in that position
        """
        if mask_average_quota is not None:
            self.mask_average_quota = mask_average_quota
        if mask_average_quota_threshold is not None:
            self.mask_average_quota_threshold = mask_average_quota_threshold

        Q = self.mask_average_quota; T = self.mask_average_quota_threshold   

        if (Q<0) or (T<0):
            
            return_average = False
        else:
            return_average = True
            
        # find the mask with the highest correlation with the main scan
        if return_average:
            valid_masks = masks[-int(len(masks)/Q):] if len(masks)>1 and len(masks)%Q==0 else masks[-int(len(masks)//Q+1):]
            
            average_mask = sum(m.astype(float) for m in valid_masks)/len(valid_masks)
            npmask = average_mask.scan

            npmask[npmask<T]=False
            npmask[npmask>=T]=True

            average_mask.scan = npmask
            return average_mask.astype(bool)


        return self.history[-1]["total_mask"]
    
    def run_checks(self, masks, composite_mask, iteration):
        if composite_mask.sum() == 0:
            print("No mask found, stopping at iteration {}".format(iteration))
            return -1
        
        # find the optimal mask so far
        latest_total_mask = sum(masks+[composite_mask]).astype("bool")

        # find the difference between the latest mask and the cumulative mask
        if self.history == []:
            difference_mask = latest_total_mask
        else:
            cumulative_masks = self.history[-1]["total_mask"]
            difference_mask = latest_total_mask ^ cumulative_masks

        num_new_dipoles = int(np.sum(difference_mask.astype("int")))
        
       
        if self.current_iteration > 15 and num_new_dipoles > self.history[-1]["N_dipoles"]*0.75:
                print("Too many new dipoles found: ({}), stopping at iteration {}".format(num_new_dipoles, iteration))
                return -3
        return latest_total_mask, difference_mask, num_new_dipoles

    def update_history(
            self, mask: List[Scan], working_scans: List[Scan], 
            Bscan: Scan, Bkernel: Scan, latest_total_mask: Scan, 
            difference_mask: Scan, num_new_dipoles: int,
            print_log: bool = True) -> None:

        if len(self.history) == 0:
            mask_history = [mask]
        else:
            mask_history = [h["mask"] for h in self.history]
            mask_history.append(mask)

        latest_total_mask = sum(mask_history).astype("bool")


        self.history.append(
            {
                "iteration": self.current_iteration,
                "N_dipoles": int(np.sum(latest_total_mask)),
                "mask": mask,
                "working_scans": working_scans,
                "binarized_scan": Bscan,
                "binarized_kernel": Bkernel,
                "total_mask": latest_total_mask,
                "difference_mask": difference_mask,
                "num_new_dipoles": num_new_dipoles,
                "damping": self.damping,
            }
        )

        if print_log:
            print("Iteration {} - N_dipoles: {}".format(self.current_iteration, np.sum(latest_total_mask.scan.astype(int))))



    def _single_component_reconstruct_from_optimum_mask(self, component=None)->DFHandler_over_Substrate:
        if component is not None:
            self.scanned_component_under_analysis = component
        
        if self.history[-1]["N_dipoles"] == 0:
            print("No Solutions, Creating empty array...")
            empty_array = self.df_handler.dipole_array.create_empty_array()
            return self.df_handler//empty_array

        r, V = self.simple_scan_analyzer.return_dipole_position_values()
        f = self.df_handler.f
        # assume all dipoles are at the same height
        if isinstance(self.df_handler.dipole_array, FlatDipoleArray):
            dipole_height = self.df_handler.dipole_array.height
        else:
            dipole_height = self.df_handler.dipole_array.r0[0,-1]
        component = self.simple_scan_analyzer.main.component

        new_flat_dipole_array = FlatDipoleArray(
            f=f, height = dipole_height, 
            r0=r, orientations=self.simple_scan_analyzer.main.component, 
            moments = V, type="Electric" if self.simple_scan_analyzer.main.field_type == "E" else "Magnetic"
            )
        new_handler = self.df_handler//new_flat_dipole_array
        new_handler.evaluate_fields()

        field_type = self.simple_scan_analyzer.main.field_type

        normMAX = getattr(new_handler, field_type).run_scan(component=component).scan.max()
        original_MAX = self.simple_scan_analyzer.main.scan.max()

        new_V = V/normMAX * original_MAX

        new_flat_dipole_array = FlatDipoleArray(
            f=f, height = dipole_height, 
            r0=r, orientations=self.simple_scan_analyzer.main.component, 
            moments = new_V, type="Electric" if self.simple_scan_analyzer.main.field_type == "E" else "Magnetic"
            )
        new_handler = self.df_handler//new_flat_dipole_array
        return new_handler
    
    def construct_dfh(self) -> DFHandler_over_Substrate:
        dipole_arrays = []
        for c in self.scanned_components:
            dfhandler = self.select_scanned_component(c)._single_component_reconstruct_from_optimum_mask()
            dipole_arrays.append(dfhandler.dipole_array)
        
        sum_darray = np.sum(dipole_arrays, axis=0)
        self.opt.dfh = dfhandler//sum_darray
        return self.opt.dfh


    
    def _return_arguments_for_optimization(self):
        dh_rec = self.construct_dfh()
        M = np.asarray(dh_rec.dipole_array.M).real[...,0]
        r0 = dh_rec.dipole_array.r0
        field_type = self.simple_scan_analyzer.main.field_type
        components = self.scanned_components
        G_matrix_name = "Gee" if field_type == "E" else "Ghh"
        G = dh_rec.G_components()[G_matrix_name][...,0]
        component_scans = [sc.main.scan for sc in self.scans]
        target = np.stack(component_scans, axis=0)
        return M, r0, G, field_type, components, target
    
    def create_optimization_object(self):
        M, r0, G, field_type, components, target = self._return_arguments_for_optimization()
        self.opt.nl_optimizer = NonLinearOptimization(M, r0,  G, target, components)
        return self.opt.nl_optimizer
    
    def lm_optimize_source_values(
            self, improve_condition_number=0, 
            L2_regularization_M=0,
            L2_regularization_P=0,
            *args, **kwargs) -> "MorphDipoleSearcher":
        nl_opt = self.create_optimization_object() 
        Mr, phr = nl_opt.regularize_G(improve_condition_number).run_lm_algorithm(
            alpha_M = L2_regularization_M, 
            alpha_P = L2_regularization_P, 
            *args, **kwargs)
        Mopt = Mr*np.exp(1j*phr)
        self.opt.M = Mopt
        fh_rec = self.construct_dfh()
        darray = fh_rec.dipole_array // Mopt[:, None]
        self.opt.dfh = DFHandler_over_Substrate(fh_rec.EM_space, fh_rec.substrate, darray)
        return self
    
    
    def to_scan(self, component=None, *args, **kwargs) -> "MorphDipoleSearcher":
        if component is None:
            component = self.simple_scan_analyzer.main.component
        if self.opt.dfh is None:
            self.construct_dfh()         
        field_type = self.simple_scan_analyzer.main.field_type
        field = getattr(self.opt.dfh.evaluate_fields(), field_type)

        # save the field to opt
        setattr(self.opt, field_type+component, field.run_scan(component))
        return getattr(self.opt, field_type+component)
    
    def save_optimum_dipole_moments_to_pickle(self, filepath):
        assert self.opt.M is not None, "No optimum dipole moments found. Run lm_optimize_source_values first."

        with open(filepath, "wb") as f:
            pickle.dump(self.opt.M, f)
        
    def plot_dipoles_number_history(self, ax: plt.Axes = None, component=None, **kwargs):
        if component is not None:
            self.scanned_component_under_analysis = component
        if len(self.scanned_components)>1:
            self.history = getattr(self, self.scanned_component_under_analysis).history

        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(5,3), constrained_layout=True)
        else:
            fig = ax.figure
        
        
        # ax.plot([h["num_new_dipoles"] for h in self.history], label="New Dipoles", marker="o", linewidth=0.9, markersize=5, linestyle=":")
        ax.plot([h["N_dipoles"] for h in self.history], label="Total Dipoles", marker="s", linewidth=0.9, markersize=5, linestyle="-", color="r")
        ax.bar(np.arange(len(self.history)), [h["num_new_dipoles"] for h in self.history], linewidth=0.5, alpha = 1, color="b")

        ax2 = ax.twinx()
        damping = [d["damping"] for d in self.history]
        ax2.bar(np.arange(len(damping)), damping, color="g", linewidth=0.5, alpha = 0.35)
        ax2.set_ylim([min(damping), 2*max(damping)])
        ax2.set_ylabel("Damping")


        ax.set_xlabel("Iterations")
        ax.set_ylabel("Number of Dipoles")

        # set the ticks to be integer numbers between the min and max
        iterations = self.N_iterations
        N_positions = 10
        xticks = np.arange(0, iterations, iterations//N_positions+1)
        xticks = set(np.insert(xticks, -1, self.N_iterations))
        ax.set_xticks(list(xticks))

        yticks = ax.get_yticks()
        yticks = np.insert(yticks, -1, self.history[-1]["N_dipoles"])
        yticks.sort()
        ax.set_yticks(yticks)
        
        
        ax.set_xlim(0, len(self.history))
        ax.set_ylim(0, max([h["N_dipoles"] for h in self.history])+1)
        # create legend outside of the plot

        totald_patch = mpatches.Patch(color='r', label='Total Dipoles')
        newd_patch = mpatches.Patch(color='b', label='New Dipoles')
        damping_patch = mpatches.Patch(color='g', label='Damping')

        labels = ["Total Dipoles", "New Dipoles", "Damping"]
        labels = [textwrap.fill(name, 10, break_long_words=False) for name in labels]

        ax.legend(
            [totald_patch, newd_patch, damping_patch], labels, 
            bbox_to_anchor=(1.10, 1.1), loc='upper left', borderaxespad=0.)

        fieldname = self.simple_scan_analyzer.main.field_type + self.simple_scan_analyzer.main.component

        ax.set_title("Number of Dipoles Found - {}".format(fieldname), fontsize=14)

        return fig, ax
    
    def plot_history(self, ax: plt.Axes = None, show_aggregate_mask=True, component=None, **kwargs):
        if component is not None:
            self.scanned_component_under_analysis = component
        if len(self.scanned_components)>1:
            self.history = getattr(self, self.scanned_component_under_analysis).history
        
        hlen = len(self.history)
        assert hlen > 0, "No history to plot"

        if ax is None:
            nrows = 1+(hlen-1)//3
            fig, ax = plt.subplots(nrows,3, figsize=(15, nrows*2.5), constrained_layout=True)
        else:
            axx = ax if ax is not isinstance(ax, (list, np.ndarray)) else ax.flat[0]
            fig = axx.figure

        mask_to_plot = "total_mask" if show_aggregate_mask else "mask"
        scatter_kwargs = {"color":"k", "s":15}
        scatter_kwargs.update(kwargs)

        for ii, h in enumerate(self.history):
            plt_objects = h["working_scans"].plot(ax=ax.flat[ii], alpha=0.8)
            plt_objects["cbar"].remove()
            my_alphas = np.ones(h["binarized_scan"].scan.shape)*0.6
            my_alphas[h["binarized_scan"].scan.T == 0] = 0
            h["binarized_scan"].plot(ax=ax.flat[ii], alpha=my_alphas, cmap="Oranges")
            h.get(mask_to_plot).scatter_plot(ax = ax.flat[ii], **scatter_kwargs)
            title = ax.flat[ii].get_title()
            ax.flat[ii].set_title(title + "_iteration_{}".format(ii))

        fig.suptitle("Fitting History", fontsize=16)
        return fig, ax

    def plot_binarized_scan(self, ax: plt.Axes = None, component=None, **kwargs):
        if component is not None:
            self.scanned_component_under_analysis = component
        if len(self.scanned_components)>1:
            self.history = getattr(self, self.scanned_component_under_analysis).history
        
        hlen = len(self.history)
        assert hlen > 0, "No history to plot"

        if ax is None:
            nrows = 1+(hlen-1)//3
            fig, ax = plt.subplots(nrows,3, figsize=(15, nrows*2.5), constrained_layout=True)
        else:
            axx = ax if ax is not isinstance(ax, (list, np.ndarray)) else ax.flat[0]
            fig = axx.figure

        bscans = [h["binarized_scan"] for h in self.history]
        bkernels = [h["binarized_kernel"] for h in self.history]

        for ii, (bscan, bkernel) in enumerate(zip(bscans, bkernels)):

            # set the scan as background
            plt_objects = self.history[ii]["working_scans"].plot(ax=ax.flat[ii], alpha=0.8)
            plt_objects["cbar"].remove()

            alpha_kernel = np.ones(bkernel.shape)*1
            alpha_kernel[bkernel.scan==False] = 0

            alpha_scan = np.ones(bscan.shape)*1
            alpha_scan[bscan.scan==False] = 0

            bscan.plot(ax=ax.flat[ii], cmap="Blues", alpha=alpha_scan.T)
            bkernel.plot(ax=ax.flat[ii], cmap="Reds", alpha=alpha_kernel.T)
            title = ax.flat[ii].get_title()
            ax.flat[ii].set_title(title + " - Iteration {}".format(ii))

            # manually create the legend
            red_patch = mpatches.Patch(color='red')
            blue_patch = mpatches.Patch(color='blue')

            ax.flat[ii].legend([red_patch, blue_patch], ["Scan", "Kernel"])
            
        
        fig.suptitle("Binarized Scans", fontsize=16)
        return fig, ax

    def animate_history(self, filepath: str = "animation.gif", component: str=None, fps=10, *args, **kwargs):        
        if component is not None:
            self.scanned_component_under_analysis = component
        if len(self.scanned_components)>1:
            self.history = getattr(self, self.scanned_component_under_analysis).history
        
        turn_to_animation(plot_working_scan)(
            self, animation_path=filepath, fps=fps, *args, **kwargs
        )
