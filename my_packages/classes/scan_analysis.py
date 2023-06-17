from dataclasses import dataclass, field
from typing import Callable, Tuple, Iterable, List, Union, Dict
from itertools import product 

import numpy as np
from skimage import morphology, filters, restoration
import matplotlib.pyplot as plt

from my_packages.classes.field_classes import Scan
from my_packages.format_checkers import check_filter_format
from my_packages.classes.dipole_fields import DFHandler_over_Substrate
from my_packages.classes.dipole_array import FlatDipoleArray


class ScanAnalyzer():
    def __init__(
            self, 
            main: Scan, 
            kernel: Scan = None, 
            erosion_function: Callable = morphology.binary_erosion, 
            resolution_shape: Tuple[int, int] = (21, 21), 
            scan_filters: List[Tuple[Callable, Tuple[any, ...]]] = [(filters.gaussian, (1))], 
            kernel_filters: List[Tuple[Callable, Tuple[any, ...]]] = [(filters.gaussian, (1))],  
            binarization_threshold: Callable | float = filters.threshold_otsu,
            threshold_object: 0 | 1 | 2 | -1 = 0,          
            ):
        self.main = main
        self.kernel = kernel
        self.erosion_function = erosion_function
        self.resolution_shape = resolution_shape
        self.scan_filters = scan_filters
        self.kernel_filters = kernel_filters
        self.binarization_threshold = binarization_threshold
        self.threshold_object = threshold_object
        self.Bkernel = None
       
        # assert that main is a Scan
        assert isinstance(self.main, Scan), "main must be a Scan"

        # Check that the kernel is a Scan or a numpy array and convert it to a Scan if it is a numpy array
        if isinstance(self.kernel, Scan) or self.kernel is None:
            pass
        elif isinstance(self.kernel, np.ndarray):
            dummy_kernel_grid = self.main.grid.resample_on_shape((3,)+self.kernel.shape)
            self.kernel = Scan(self.kernel, dummy_kernel_grid, self.main.f, self.main.axis, self.main.component, self.main.field_type)
        else:
            raise ValueError("kernel must be a Scan or a numpy array")
        
        # Check that the resolution shape is a tuple of two integers
        assert isinstance(self.resolution_shape, tuple), "resolution_shape must be a tuple"
        self.variations_for_composite_mask = None



    @property
    def erosion_function(self):
        return self._erosion_function 

    @erosion_function.setter
    def erosion_function(self, erosion_function):
        self._erosion_function = ScanAnalyzer.check_filter_format(erosion_function)

    @property
    def scan_filters(self):
        return self._scan_filters
    
    @scan_filters.setter
    def scan_filters(self, scan_filters):
        self._scan_filters = ScanAnalyzer.check_filter_format(scan_filters)
    
    @property
    def kernel_filters(self):
        return self._kernel_filters
    
    @kernel_filters.setter
    def kernel_filters(self, kernel_filters):
        self._kernel_filters = ScanAnalyzer.check_filter_format(kernel_filters)
    @property
    def binarization_threshold(self):
        return self._binarization_threshold
    @binarization_threshold.setter
    def binarization_threshold(self, binarization_threshold):
        if isinstance(binarization_threshold, float):
            binarization_threshold = lambda x: binarization_threshold
        self._binarization_threshold = ScanAnalyzer.check_filter_format(binarization_threshold)


    @staticmethod
    def check_filter_format(filters, verbose=False) -> None:
        return check_filter_format(filters, verbose)
    
    def get_kernel_from_dh(
            self, fh_handler: DFHandler_over_Substrate, 
            probe_height: float, kernel_shape: Tuple[int, int] = None) -> "ScanAnalyzer":
        self.kernel = self._find_kernel_from_data_handler(fh_handler, probe_height, kernel_shape)
        return self

    def _find_kernel_from_data_handler(self, fh_handler: DFHandler_over_Substrate, probe_height: float =0, kernel_shape: Tuple[int, int] = None):
        if kernel_shape is None:
            kernel_shape = self.main.shape
        
        # find kernel
        orientation = self.main.component
        field_type = "Electric" if self.main.field_type == "E" else "Magnetic"
        handler = fh_handler.get_handler_for_single_dipole(orientation, field_type).evaluate_fields()
        kernel = getattr(handler, self.main.field_type).run_scan(axis=self.main.axis, component=self.main.component, index=probe_height)
        print(kernel_shape)
        kernel = kernel.resample_on_shape(kernel_shape).normalize()  
        return kernel


    def binarize_and_erode(
            self, 
            threshold_object: 0 | 1 | 2 | -1 = None,
            apply_shared_filters: List | Tuple | Callable | None = None,
            kernel: Union[np.ndarray, "Scan"] = None, 
            erosion_func: Callable | Tuple[Callable, List | Dict] = None, 
            thresholding_func: Callable | Tuple[Callable, List | Dict] = None, 
            verbose=False,
            attribute_name = "mask",
            default_threshold_value = 0.5,
            Bkernel: Scan = None
            )->"ScanAnalyzer":
            """Apply an erosion function to the scan

            Parameters
            ----------
            erosion_func : function that takes a scan and a kernel and returns an eroded scan

            kernel : kernel to use for erosion

            thresholding_func : function that takes a scan and returns a threshold value

            threshold_object : 0 | 1 | 2, optional; by default 1.
            0 indicates that the threshold value should be obtained from the scan itself,
            1 indicates that the threshold value should be obtained from the kernel,
            2 indicates that the threshold for the scan should be obtained from the scan and 
            the threshold for the kernel should be obtained from the kernel
            -1 indicates that that the scan should be thresholed, while the previous binarized kernel should be used

            self.resolution_shape : tuple or list, optional; by default None.
            If not None, the scan will be resampled to the given shape before applying the erosion function
            if a list is passed, the list should be comprised of two tuples, the first tuple should contain the
            shape of the scan and the second tuple should contain the shape of the kernel. If a tuple is passed,
            the same shape will be used for both the scan and the kernel.

            verbose : bool, optional; by default False. If True, print statements will be printed to the console
            as the pipeline is running.

            apply_filters : list of functions, optional; by default None. If not None, the list of functions will be
            applied to the scan and the kernel before the erosion function is applied. If a tuple is passed, the
            first element of the tuple should be the function and the second element should be a list of arguments.
            If a dict is passed, the dict will be passed as keyword arguments to the function.

            Returns
            -------
            Scan
                eroded scan

            """

            def mprint(*args):
                if verbose:
                    print(*args)
            if threshold_object is None:
                threshold_object = self.threshold_object
            if kernel is not None:
                self.kernel = kernel
            if erosion_func is not None:
                self.erosion_function = erosion_func
            if thresholding_func is not None:
                self.binarization_threshold = thresholding_func
            
            # filters
            if apply_shared_filters is not None:
                apply_shared_filters = ScanAnalyzer.check_filter_format(apply_shared_filters)
                self.scan_filters += apply_shared_filters
                self.kernel_filters += apply_shared_filters
            
            # apply filters and normalize
            myscan = self.main.apply_pipeline(self.scan_filters).normalize()
            my_kernel = self.kernel.apply_pipeline(self.kernel_filters).normalize()


            # resample
            if isinstance(self.resolution_shape, Tuple):
                myscan = myscan.resample_on_shape(self.resolution_shape)
                my_kernel = my_kernel.resample_on_shape(self.resolution_shape)
            elif isinstance(self.resolution_shape, list):
                myscan = myscan.resample_on_shape(self.resolution_shape[0])
                my_kernel = my_kernel.resample_on_shape(self.resolution_shape[1])
            else:
                raise ValueError("resolution_shape must be a tuple or a list")
            
            # the normalization is done above, so in the find threshold step we don't need to do it again
            if threshold_object == 0 or threshold_object == -1:
                try:
                    scan_norm_threashold = myscan.apply_pipeline(self.binarization_threshold, include_normalization_step =False, return_raw = True)
                except RuntimeError as e:
                    print("Could not find threshold for scan, using default value of 0.5")
                    print(e)
                    scan_norm_threashold = default_threshold_value            
                kernel_threshold_norm = scan_norm_threashold
            elif threshold_object == 1:
                kernel_threshold_norm = my_kernel.apply_pipeline(self.binarization_threshold, include_normalization_step =False, return_raw = True)
                scan_norm_threashold = kernel_threshold_norm
            elif threshold_object == 2:
                scan_norm_threashold = myscan.apply_pipeline(self.binarization_threshold, include_normalization_step =False, return_raw = True)
                kernel_threshold_norm = my_kernel.apply_pipeline(self.binarization_threshold, include_normalization_step =False, return_raw = True)

            self.kernel_threshold_norm = kernel_threshold_norm
            self.scan_threshold_norm = scan_norm_threashold  
            
            # for the verbose option we print the thresholds
            mprint("field scan threshold: ", scan_norm_threashold)
            mprint("kernel threshold: ", kernel_threshold_norm)

            # binarize 
            binarized_scan = myscan.binarize(scan_norm_threashold)
            
            if threshold_object == -1 and Bkernel is not None:
                binarized_kernel_scan = Bkernel
            else:
                binarized_kernel_scan = my_kernel.binarize(kernel_threshold_norm)

            # clip the edges of the binarized scan to zero
            
            # pad the trimmed array with zeros
            pad_width = 1
            padded_scan = np.pad(binarized_scan.scan[pad_width:-pad_width, pad_width:-pad_width:], pad_width=pad_width, mode='constant', constant_values=0)
            binarized_scan = binarized_scan.update_with_scan(padded_scan)
           

            self.Bscan = binarized_scan
            self.Bkernel = binarized_kernel_scan

            # erode
            # binarized_kernel_scan.plot()
            mask = binarized_scan.apply_pipeline(self.erosion_function, binarized_kernel_scan.scan)
            setattr(self, attribute_name, mask)
            return self
    
    def _check_if_mask(self, mask: Union[Scan, np.ndarray]):
        if mask is not None:       

            # convert from numpy array to Scan
            if isinstance(mask, Scan):
                pass
            elif isinstance(mask, np.ndarray):
                assert mask.ndim == 2, "mask must be a 2D array"
                mask = Scan(
                    self.main.scan, self.main.grid, 
                    self.main.f, self.main.axis, self.main.component,  
                    self.main.field_type
                    ).resample_on_shape(mask.shape
                    ).update_with_scan(mask)
            else:
                raise ValueError("mask must be a Scan or a numpy array")
            
            # check the scan dtype
            if mask.dtype == int:
                assert mask.scan.max() <= 1 and mask.scan.max() >= 0, "mask must be a boolean array"
            else:
                assert mask.dtype == bool, "mask must be a boolean array"
            
            return mask
        else:
            assert hasattr(self, "mask"), "mask must be a boolean array"
            return self.mask
    
    def return_dipole_position_values(self, mask: Union[Scan, np.ndarray] = None, average_filter=None, clip_quota=0, min_amplitude=0):
        mask = self._check_if_mask(mask)
        x_i, x_j = np.where(mask)
        x = mask.grid.x[x_i]
        y = mask.grid.y[x_j]
        r0 = np.stack([x,y], axis=1)
        scanned_field = self.main.resample_on_shape(mask.shape).normalize()

        if average_filter is not None:
            assert average_filter % 2 == 1, "average_filter must be an odd number"
            padded_field = np.pad(scanned_field.scan, average_filter//2, mode='reflect')
            field_values = []
            for i,j in zip(x_i, x_j):
                values_under_av_filter = padded_field[
                    i-average_filter//2:i+average_filter//2,
                    j-average_filter//2:j+average_filter//2
                    ]
                field_values.append(np.mean(values_under_av_filter))
        else:
            field_values = [scanned_field.scan[i, j] for i,j in zip(x_i, x_j)]

        # lower_i = x_i-average_filter//2;  upper_i= x_i+average_filter//2
        # lower_j = x_j-average_filter//2;  upper_j= x_j+average_filter//2
        

        # field_values = [np.mean(scanned_field.scan[i:j, k:l]) for i,j,k,l in zip(lower_i, upper_i, lower_j, upper_j)]

        field_values = np.nan_to_num(field_values)
        max_fv = np.max(field_values)
        min_fv = np.min(field_values)

       

        V = []

        if max_fv == min_fv:
            V = field_values
        else:
            for fv in field_values:
                if fv > clip_quota*max_fv:
                    V.append(fv)
                else:
                    V.append(min_amplitude)
        return r0, V


    def extract_composite_mask(
            self,
            variable_parameter: Dict[str, List[any]], 
            method: str = "average", 
            save_individual_masks: bool = False,
            attribute_name: str = "mask",
            basis_scan: Scan = None,
            Bkernel: Scan = None, verbose=False, 
            )->"ScanAnalyzer":
        mydict = {
            0: "average",
            1: "max",
            2: "min"
        }
        def mprint(*args):
            if verbose:
                print(*args)

        if method in mydict.keys():
            pass
        elif method in mydict.values():
            # find the key for the value
            method = list(mydict.keys())[list(mydict.values()).index(method)]
        else:
            raise ValueError("method must be one of {} or {}. You inserted: ".format(mydict.values(), mydict.keys(), method))
        
        mprint("the selected method is {}".format(mydict[method]))

        # make sure that all the values are iterable
        for k in variable_parameter.keys():
            if not isinstance(variable_parameter[k], list):
                variable_parameter[k] = [variable_parameter[k]]
        
        mprint("saving the parameter varaiations as attribute")
        self.variations_for_composite_mask = variable_parameter
        
        # create a list of dictionaries with all the possible combinations of the parameters
        list_of_updates = [dict(zip(variable_parameter.keys(), v)) for v in product(*variable_parameter.values())]        

        # create a list of eroded masks
        eroded_masks = []

        for param_update in list_of_updates:
            # separate the parameters that are not attributes of the class
            threshold_object = param_update.pop("threshold_object", self.threshold_object)
            apply_shared_filters = param_update.pop("apply_shared_filters", None)
            
            # update the attributes of the class
            self.update_from_dict(param_update)
            maskname = "mask_{}".format("_".join([str(v) for v in param_update.values()])) if save_individual_masks else attribute_name
            eroded_masks.append(self.binarize_and_erode(
                threshold_object=threshold_object, apply_shared_filters=apply_shared_filters, attribute_name=maskname, Bkernel=Bkernel
                ).mask)
            
            # verbose
            key = list(param_update.keys())[0]
            mprint("eroded mask for {} = {}".format(key, param_update[key]))
        
        new_grid = eroded_masks[0].grid.copy()
        eroded_masks = np.asarray(eroded_masks, dtype=int)

        if method == 0:
            # average
            final_mask = np.average(eroded_masks, axis=0)
            n_trials = len(eroded_masks)
            final_mask = np.where(final_mask > 1/2, 1, 0)
            mprint("averaged mask with n_trials = {}".format(n_trials))
        elif method == 1:
            # max
            final_mask = np.max(eroded_masks, axis=0)
            mprint("max mask")
        elif method == 2:
            # min
            final_mask = np.min(eroded_masks, axis=0)
            mprint("min mask")
        else:
            raise ValueError("method must be one of {} or {}".format(mydict.values(), mydict.keys()))
        mask = Scan(final_mask, new_grid, self.main.f, self.main.axis, self.main.component,  self.main.field_type)
        setattr(self, attribute_name, mask)
        return self
    
    def update_from_dict(self, mydict)-> None:
        assert isinstance(mydict, dict)
        methods = [k if k[0]!="_" else k[1:] for k in self.__dict__.keys()]
        assert all([k in methods for k in mydict.keys()])
        for k in mydict.keys():
            setattr(self, k, mydict[k])

    def plot_overlay(self, maskname: str = "mask", mask: Scan = None, ax=None, background_kwargs: dict = {}, **kwargs):
        if mask is None:
            if not hasattr(self, maskname):
                raise ValueError("the class has no attribute called {}".format(maskname))
            mask = getattr(self, maskname)
        else:
            mask = self._check_if_mask(mask)
                
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True, dpi=100)  

        background_kwargs = { "alpha": 1, "antialiased": False } | background_kwargs

        # background plot
        self.main.plot(ax=ax, **background_kwargs)

        if self.main.field_type == "E" and self.main.component=="z":
            default_marker = "o"
        elif self.main.field_type=="H" and self.main.component=="y":
            default_marker = "^"
        elif self.main.field_type=="H" and self.main.component=="x":
            default_marker = ">"
        else:
            default_marker = "x"

        scatter_kwargs = {
            "marker": default_marker,
            "color": "k",
            "s": 20
        }

        scatter_kwargs.update(kwargs)

        # assume xy plane
        xvals = mask.grid.x[np.where(mask.data)[0]]
        yvals = mask.grid.y[np.where(mask.data)[1]]
        ax.scatter(xvals, yvals, **scatter_kwargs)
        return ax
    
    