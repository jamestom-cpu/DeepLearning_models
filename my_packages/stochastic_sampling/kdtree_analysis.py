import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from matplotlib import ticker

from .helper_functions.kdtree_funcs import get_kdtree_partitions
from .helper_functions.plotting_funcs import draw_rectangles_and_points, streatch_boundaries



class KDTree_Analysis():
    def __init__(self, points, xbounds, ybounds, leafsize):
        self.points = points
        self.xbounds = xbounds
        self.ybounds = ybounds
        self.leafsize = leafsize
        self.rectangles = None
        self.indices = None
        self._get_kdtree_partitions()
    
    def _get_kdtree_partitions(self):
        self.rectangles, self.indices = get_kdtree_partitions(self.points, self.leafsize, bounds=[self.xbounds, self.ybounds])
        return self
    
    @property
    def leaf_areas(self):
        if self.rectangles is None:
            return None
        return np.asarray([np.abs(rect[1]-rect[0]) * np.abs(rect[3]-rect[2]) for rect in self.rectangles])

    @ property
    def leaf_dict(self): 
        return {key: self.indices[key] for key in range(len(self.rectangles))}
    
    def highlight_leaf(self, leaf_index, ax=None, scatter_kwargs={}, patch_kwargs={}):
        leaf = self.rectangles[leaf_index]  
        points = self.points[self.indices[leaf_index]]
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.set_xlim(streatch_boundaries(self.xbounds, 0.1))
            ax.set_ylim(streatch_boundaries(self.ybounds, 0.1))
        
        my_patch_kwargs = dict(edgecolor='k', facecolor='b', linewidth=3, alpha=0.6) | patch_kwargs
        my_scatter_kwargs=dict(c='r', s=100, edgecolors='k', linewidths=1) | scatter_kwargs
        draw_rectangles_and_points([leaf], points, ax=ax, scatter_kwargs=my_scatter_kwargs, patch_kwargs=my_patch_kwargs);

        ax.set_label('x [mm]')
        ax.set_label('y [mm]')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*1000:.1f}'))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*1000:.1f}'))
        return self

    
    def plot(self, ax=None, title=None, scatter_kwargs={}, patch_kwargs={}):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 3))
        
        draw_rectangles_and_points(self.rectangles, self.points, ax=ax, scatter_kwargs=scatter_kwargs, patch_kwargs=patch_kwargs);
        
        # change units to mm
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*1000:.1f}'))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*1000:.1f}'))

        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        default_title = f'KDTree with leafsize={self.leafsize}'
        my_title = " - ".join([title, default_title]) if title else default_title
        ax.set_title(my_title, fontsize=14)
        return self