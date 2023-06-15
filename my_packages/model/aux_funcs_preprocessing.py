import numpy as np
from scipy.ndimage import generic_filter

def local_outlier_removal(data, window_size=3, z_score_threshold=3):
    # function to apply to each window
    def func(window):
        median = np.median(window)
        abs_diff = np.abs(window - median)
        mad = np.median(abs_diff)
        modified_z_scores = 0.6745 * abs_diff / mad
        window[modified_z_scores > z_score_threshold] = median
        return window[-1]

    return generic_filter(data, func, size=window_size)