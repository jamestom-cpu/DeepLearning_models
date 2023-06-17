from typing import Tuple
import numpy as np

from .voronoi import VoronoiAnalysis
from .kdtree_analysis import KDTree_Analysis

class SpatialAnalysis():
    def __init__(
            self, 
            points: np.ndarray, 
            xbounds: Tuple,
            ybounds: Tuple,
            default_leafsize: int = 3,
            seed: int = None,
            include_default_voronoi: bool = True 
            ):
        self.points = points
        self.xbounds = xbounds
        self.ybounds = ybounds
        self.default_leafsize = default_leafsize
        self.seed = seed
        if include_default_voronoi:
            self.Vor = VoronoiAnalysis(self.points, self.xbounds, self.ybounds, self.seed)
        self.KDTree = self._startKDTree
    
    def _startKDTree(self, leafsize=None):
        if leafsize is None:
            leafsize = self.default_leafsize
        self._KDTree = KDTree_Analysis(self.points,
                                        xbounds=self.xbounds,
                                        ybounds=self.ybounds,
                                        leafsize=leafsize)
        return self._KDTree
        
    
        
    
        