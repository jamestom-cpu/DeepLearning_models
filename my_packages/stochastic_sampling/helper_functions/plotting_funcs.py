from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
from shapely.geometry import Polygon
from matplotlib import patches

def streatch_boundaries(bounds: Tuple, factor: float):
    min, max = bounds
    delta = max - min
    return (min - delta * factor, max + delta * factor)

def draw_patch(rect, ax:Axes, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    x_min, x_max, y_min, y_max = rect
    ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, **kwargs))

def draw_rectangles_and_points(rectangles, points, ax=None, scatter_kwargs={}, patch_kwargs={}):
    # Create the plot.
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))

    # Draw the points.
    ax.scatter(points[:, 0], points[:, 1], **scatter_kwargs)

    my_kwargs = dict(edgecolor='k', facecolor='none', linewidth=2)
    my_kwargs.update(patch_kwargs)
    # Draw the rectangles.
    [draw_patch(rect, ax, **my_kwargs) for rect in rectangles]
    return ax 

from matplotlib import patches


def plot_polygon(polygon:Polygon, ax=None, fill=None, alpha_fill=1.0, include_area_value=False, **kwargs):
        # Get the polygon
        # Create a figure and axes
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        polygon_vertices = np.asarray(polygon.exterior.coords.xy).T

        my_kwargs = {"color":"k", "linewidth":2, "alpha":0.6}
        my_kwargs.update(kwargs)  
        ax.plot(*polygon_vertices.T, **my_kwargs)  

        # If a fill color was specified, fill in the polygon
        if fill is not None:
            polygon_patch = patches.Polygon(polygon_vertices, fill=True, color=fill, alpha=alpha_fill)
            ax.add_patch(polygon_patch)
        
        if include_area_value:
            centroid = np.mean(polygon_vertices, axis=0)
            # Annotate the plot with the area of the polygon at the centroid
            ax.text(centroid[0], centroid[1], f'Area: {polygon.volume:.2f}', horizontalalignment='center')
        return ax
        