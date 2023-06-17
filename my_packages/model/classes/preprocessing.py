from scipy.ndimage import uniform_filter, gaussian_filter, median_filter
from copy import copy

from my_packages.classes.field_classes import Scan
from my_packages.model.aux_funcs_preprocessing import local_outlier_removal

class MeasurementPreprocessing():
    def __init__(
            self, 
            scan: Scan, 
            window_size_outlier_removal=3, 
            z_score_threshold=3,
            window_size_smoothing=3,
            sigma_smoothing=1,
            ):
        self.original_scan = copy(scan)
        self.scan = scan
        self.window_size_outlier_removal = window_size_outlier_removal
        self.z_score_threshold = z_score_threshold
        self.window_size_smoothing = window_size_smoothing
        self.sigma_smoothing = sigma_smoothing

    
    def remove_outliers(self):
        cleaned_data = local_outlier_removal(self.scan.scan, window_size=self.window_size_outlier_removal, z_score_threshold=self.z_score_threshold)
        self.scan = self.scan.update_with_scan(cleaned_data)
        return self
    
    def smooth(self, method="uniform", window=None, sigma=None):
        window = self.window_size_smoothing if window is None else window
        sigma = self.sigma_smoothing if sigma is None else sigma

        if method == "uniform":
            smoothed_data = uniform_filter(self.scan.scan, size=window, mode="reflect")
        elif method == "gaussian":
            smoothed_data = gaussian_filter(self.scan.scan, sigma=sigma, mode="reflect")
        elif method == "median":
            smoothed_data = median_filter(self.scan.scan, size=window, mode="reflect")
        else:
            raise ValueError("method must be one of 'uniform', 'gaussian' or 'median'")
        self.scan = self.scan.update_with_scan(smoothed_data)
        return self
    
    def reset_scan(self):
        self.scan = copy(self.original_scan)
        return self