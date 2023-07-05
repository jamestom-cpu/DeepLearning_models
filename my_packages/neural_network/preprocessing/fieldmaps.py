import numpy as np
import scipy.ndimage as ndi

class FieldmapPreprocessor:
    def __init__(self, fieldmap, sigma=1.0):
        self.fieldmap = fieldmap
        self.sigma = sigma
        self.min = np.min(self.fieldmap)
        self.max = np.max(self.fieldmap)

    def remove_nan(self):
        nan_elements = np.isnan(self.fieldmap)
        if np.any(nan_elements):
            self.fieldmap[nan_elements] = np.nanmean(self.fieldmap)

    def gaussian_blur(self, sigma=None):
        my_sigma = self.sigma if sigma is None else sigma
        self.fieldmap = ndi.gaussian_filter(self.fieldmap, sigma=my_sigma)

    def normalize(self):
        self.fieldmap = (self.fieldmap - np.min(self.fieldmap)) / (np.max(self.fieldmap) - np.min(self.fieldmap))

    def undo_normalization(self):
        self.fieldmap = self.fieldmap * (self.max - self.min) + self.min 

    def preprocess(self, sigma=None):
        self.remove_nan()
        self.gaussian_blur(sigma=sigma)
        # self.normalize()
        return self.fieldmap