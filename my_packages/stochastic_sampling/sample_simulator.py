from typing import Tuple
from collections import namedtuple
import numpy as np

from my_packages.classes.field_classes import Field3D

# define Point as a numpy array containing 3 float values
Point = namedtuple('Point', ['x', 'y', 'z'])

class SampleSimulator():
    def __init__(
            self, 
            field: Field3D,
            frequency: float = None,
            xbounds: Tuple = None,
            ybounds: Tuple = None,
            zbounds: Tuple = None,
            component: str = None
            ): 
        self.freq = frequency
        self.F = field
        self.xbounds = self.F.grid.bounds()[0] if xbounds is None else xbounds
        self.ybounds = self.F.grid.bounds()[1] if ybounds is None else ybounds
        self.zbounds = self.F.grid.bounds()[2] if zbounds is None else zbounds
        self.component = component
        self.frequency = frequency if frequency is not None else self.F.freqs[0]
    
    def _take_component(self, val: np.ndarray):
        if self.component is None:
            return val
        else:
            lookup = {'x': 0, 'y': 1, 'z': 2}
            return val[lookup[self.component], ...]
    
    def _consider_field_at_frequency(self):
        return self.F.evaluate_at_frequencies(self.frequency, overwrite=False) 

    def _transform_to_Point(self, point: np.ndarray):
        return Point(point[0], point[1], point[2])

    def interpolated_sample(self, point: Point):
        if not isinstance(point, Point):
            point = self._transform_to_Point(point)
        field_at_freq = self._consider_field_at_frequency()
        interpolated_measurement = field_at_freq.evaluate_at_points([[point.x, point.y, point.z]])[0]
        return self._take_component(np.abs(interpolated_measurement))
        