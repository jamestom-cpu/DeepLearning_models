from typing import Tuple, Iterable, Dict
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import norm, gamma, lognorm, uniform, kstest
from scipy.spatial import Voronoi 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import matplotlib.patches as patches
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

from .helper_functions.voronoi_funcs import *

# Constants
FIG_SIZE = (8, 4)

# Generate random data points
class VoronoiAnalysis():
    def __init__(self, points: np.ndarray, xbounds: Tuple, ybounds: Tuple, seed: int = None):
        assert isinstance(points, np.ndarray), "Points must be a numpy array"
        assert points.shape[1] == 2, "Points must be 2D"
        
        self.seed = seed
        if seed is not None:
            np.random.seed(self.seed)

        self.points = points
        self.xbounds = np.asarray(xbounds)
        self.ybounds = np.asarray(ybounds)
        self.vor = Voronoi(points)
        self.n_samples = len(points)
        self._construct_closed_polygons()
        self.areas = self._calculate_voronoi_areas()
        self.skewness = self._calculate_voronoi_skewness()
        self.scanning_area = np.ptp(self.xbounds) * np.ptp(self.ybounds)
        ratio_area_covered = np.round((sum(self.areas) / self.scanning_area), 3)
        if ratio_area_covered != 1.0:
            print(
                "portion of scanning area covered by voronoi polygons: {:.1f} %".format(ratio_area_covered*100)
                )
            
        self.norm_areas = self.areas / self.scanning_area
        self.df = df = pd.DataFrame({
            'Area': self.norm_areas,
            'Skewness': self.skewness
        })
        
    
    def _construct_closed_polygons(self):
        self.polygons = get_polygons(self.vor, self.xbounds, self.ybounds)


    def _calculate_voronoi_areas(self):
        return [p.volume for p in self.polygons]
    
    def _calculate_voronoi_skewness(self):
        return [self._polygon_skewness(p) for p in self.polygons]
    
    @staticmethod
    def _polygon_skewness(hull):
        vertices = hull.points[hull.vertices]
        center_of_mass = np.mean(vertices, axis=0)
        distances = cdist([center_of_mass], vertices)[0]
        return np.max(distances) / np.min(distances)
    
    def find_k_nearest_neighbors(self, index: int, k: int)->Dict:
        # Get the coordinates of the selected Voronoi cell
        cell_coords = self.points[index]

        # Build a KDTree using all points except the selected cell
        tree = cKDTree(np.delete(self.points, index, axis=0))

        # Query the KDTree to find the nearest neighbors
        distances, neighbor_indices = tree.query(cell_coords, k=k)

        # Adjust neighbor indices to account for the removed cell
        neighbor_indices = np.where(neighbor_indices >= index, neighbor_indices + 1, neighbor_indices)

        return {"distances": distances, "indices": neighbor_indices}
    
    def find_adjacent_neighbors(self, index):
        adjacent_indices = find_voronoi_cell_adjacent_neighbors(self.vor, index, self.xbounds, self.ybounds)
        cell_distances = np.linalg.norm(self.points[adjacent_indices] - self.points[index], axis=1)
        return {"indices": adjacent_indices, "distances": cell_distances}
    
    
    def plot_polygon_list(self, indices: Iterable[int], ax=None, show_samples = True, marker_kwargs = {}, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlim(self.xbounds*1.05)
            ax.set_ylim(self.ybounds*1.05)
            ax.set_aspect('equal')
            ax.set_xlabel('x [mm]')
            ax.set_ylabel('y [mm]')
            formatter = ticker.FuncFormatter(ticker_meters_to_mm)
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            ax.set_title("Voronoi Tesselation")

        my_kwargs = {"color":"k", "linewidth":2, "alpha":0.6, "fill":"b", 
                     "alpha_fill":0.2, "include_area_value":False}
        my_kwargs.update(kwargs)

        for index in indices:
            self.plot_polygon(index, ax=ax, show_sample=show_samples, marker_kwargs=marker_kwargs, **my_kwargs)


    def get_polygon(self, index: int):
        return self.polygons[index]
    
    
    def plot_all_polygons(self, ax=None, show_samples=True, title="Voronoi Tesselation", marker_kwargs={}, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlim(self.xbounds*1.05)
            ax.set_ylim(self.ybounds*1.05)
            ax.set_aspect('equal')
            ax.set_xlabel('x [mm]')
            ax.set_ylabel('y [mm]')
            formatter = ticker.FuncFormatter(ticker_meters_to_mm)
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            ax.set_title(title)

        my_kwargs = {"color":"k", "linewidth":2, "alpha":0.6, "fill":"b", 
                     "alpha_fill":0.2, "include_area_value":False}
        my_kwargs.update(kwargs)

        for index in range(len(self.polygons)):
            self.plot_polygon(index, ax=ax, show_sample=show_samples, marker_kwargs=marker_kwargs, **my_kwargs)

        # if show_samples:
        #     ax.scatter(self.points[:, 0], self.points[:, 1], s=10, c="r")
        
    
    def summary_statistics(self):
        df = pd.DataFrame({
            'Area': self.norm_areas,
            'Skewness': self.skewness
        })
        return df.describe()
    
    def quality_of_gamma_fit(self, quantity="areas"):
        if quantity == "areas":
            var = self.norm_areas
            floc = 0.0
        else:
            var = self.skewness
            floc=0.99
        
        # Estimate the parameters of the gamma distribution
        shape, loc, scale = gamma.fit(var, floc=floc)
        # Perform the Kolmogorov-Smirnov test
        D, p = kstest(var, 'gamma', args=(shape, loc, scale))

        # Return the estimated parameters, the D statistic, and the p-value
        return {"D statistic": D, "p-value": p, "params": (shape, loc, scale)}

    def quality_of_lognormal_fit(self, quantity="areas"):   
        # Estimate the parameters of the lognormal distribution
        if quantity == "areas":
            var = self.norm_areas
            floc = 0.0
        else:
            var = self.skewness
            floc = 0.99
        
        shape, loc, scale = lognorm.fit(var, floc=floc)

        # Perform the Kolmogorov-Smirnov test
        D, p = kstest(var, 'lognorm', args=(shape, loc, scale))   

        # Return the estimated parameters, the D statistic, and the p-value
        return {"D statistic": D, "p-value": p, "params": (shape, loc, scale)}
    
    def quality_of_uniform_fit(self, quantity="areas"):     
            if quantity == "areas":
                var = self.norm_areas
            else:
                var = self.skewness
            
            # Estimate the parameters of the uniform distribution
            loc, scale = uniform.fit(var)
    
            # Perform the Kolmogorov-Smirnov test
            D, p = kstest(var, 'uniform', args=(loc, scale))
    
            # Return the estimated parameters, the D statistic, and the p-value
            return {"D statistic": D, "p-value": p, "params": (loc, scale)}
    
    def quality_of_normal_fit(self, quantity="areas"):
        if quantity == "areas":
            var = self.norm_areas
            known_mean = 1/self.n_samples
            var_corr = 1.0
        else:
            var = self.skewness
            known_mean = np.sum(self.skewness)/self.n_samples
            var_corr = self.n_samples/(self.n_samples-1)

        # Estimate the variance using MLE
        estimated_variance = np.mean((var - known_mean)**2)*var_corr
        estimated_std = np.sqrt(estimated_variance)
        
        # Standardize the data
        standardized_var = (var - known_mean) / estimated_std

        # Perform the Kolmogorov-Smirnov test
        D, p = kstest(standardized_var, 'norm')

        # Return the estimated variance, the D statistic, and the p-value
        return {"D statistic": D, "p-value": p, "params": (known_mean, estimated_std)}

    def test_best_model(self, quantity="areas"):
        normal_fit = self.quality_of_normal_fit(quantity) | {"Distribution": "normal"}
        gamma_fit = self.quality_of_gamma_fit(quantity) | {"Distribution" : "gamma"}
        lognormal_fit = self.quality_of_lognormal_fit(quantity) | {"Distribution": "lognorm"}
        uniform_fit = self.quality_of_uniform_fit(quantity) | {"Distribution": "uniform"}

        # Find the best-fit model
        best_model = max([normal_fit, gamma_fit, lognormal_fit, uniform_fit], key=lambda x: x["p-value"])
        return best_model
    
    def plot_best_model(self, best_model, quantity="areas", ax=None, **kwargs):
        if quantity == "areas":
            x = np.linspace(0, self.df["Area"].max()*1.05, 100)
        else: 
            x = np.linspace(1, self.df["Skewness"].max()*1.05, 100)

        # Plot the PDF of the best-fit model
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,4))
        
        if best_model["Distribution"] == "normal":
            pdf = norm.pdf(x, *best_model["params"])
            ax.plot(x, pdf, label="Best-fit Normal", **kwargs)

        elif best_model["Distribution"] == "gamma":
            shape, loc, scale = best_model["params"]
            pdf = gamma.pdf(x, shape, loc=loc, scale=scale)
            ax.plot(x, pdf, label="Best-fit Gamma", **kwargs)

        elif best_model["Distribution"] == "lognorm":
            shape, loc, scale = best_model["params"]
            pdf = lognorm.pdf(x, shape, loc=loc, scale=scale)
            ax.plot(x, pdf, label="Best-fit Log-normal", **kwargs)
        
        elif best_model["Distribution"] == "uniform":
            loc, scale = best_model["params"]
            pdf = uniform.pdf(x, loc=loc, scale=scale)
            ax.plot(x, pdf, label="Best-fit Uniform", **kwargs)

    def plot_histograms(self, ax=None, include_model_fitting=False, **kwargs):
        df = self.df

        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=FIG_SIZE)

        my_kwargs = {}
        my_kwargs.update(kwargs)

        sns.histplot(df["Area"], ax=ax[0], stat="density", **my_kwargs)
        sns.histplot(df["Skewness"], ax=ax[1], stat="density", **my_kwargs)


        if not include_model_fitting:
            sns.kdeplot(df["Area"], ax=ax[0], alpha=0.5, linewidth=2)
            sns.kdeplot(df["Skewness"], ax=ax[1], alpha=0.5, linewidth=2)

        ticker_percent = lambda x, pos: "{:.2f}%".format(x*100)
        xformatter = ticker.FuncFormatter(ticker_percent)


        ax[0].set_xlabel("Area fraction")
        ax[0].set_ylabel("Density")
        ax[0].xaxis.set_major_formatter(xformatter)

        ax[1].set_xlabel("Skewness")
        ax[1].set_ylabel("Density")

        if include_model_fitting:
            # normal_fit = self.quality_of_normal_fit() | {"Distribution": "normal"}
            # gamma_fit = self.quality_of_gamma_fit() | {"Distribution" : "gamma"}
            # lognormal_fit = self.quality_of_lognormal_fit() | {"Distribution": "lognorm"}

            # # Find the best-fit model
            # best_model = max([normal_fit, gamma_fit, lognormal_fit], key=lambda x: x["p-value"])

            best_model_areas = self.test_best_model("areas")
            best_model_sk = self.test_best_model("sk")
            
            plotting_params = {
                "color": "r",
                "linestyle": "-.",
                "linewidth": 1,
                "alpha": 0.75
            }

            self.plot_best_model(best_model_areas, ax=ax[0], quantity="areas", **plotting_params)
            self.plot_best_model(best_model_sk, ax=ax[1], quantity="sk", **plotting_params)
            
            pval_label_area = "p-value: {:.2f}".format(best_model_areas["p-value"])
            pval_label_sk = "p-value: {:.2f}".format(best_model_sk["p-value"])
       
            var_label_area = "std: {:.2e}".format(best_model_areas["params"][-1])
            var_label_sk = "std: {:.2e}".format(best_model_sk["params"][-1])

            # ax[0].text(0.65, 0.75, pval_label_area , transform=ax[0].transAxes)
            # ax[1].text(0.65, 0.75, pval_label_sk, transform=ax[1].transAxes)

            

            original_legend = ax[1].legend().get_texts()[0].get_text()
            ax[1].legend(["\n".join([original_legend,pval_label_sk, var_label_sk])])

            original_legend = ax[0].legend().get_texts()[0].get_text()
            ax[0].legend(["\n".join([original_legend,pval_label_area, var_label_area])])
        return ax
    

    
    def bar_plot_3D(self, bar_width, color="b", **kwargs): 
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection='3d')
        total_area = sum(self.areas)
        z = np.array(self.areas) / total_area
        color = color
        for i in range(len(self.points)):
            ax.bar3d(self.points[i, 0]*1e3, self.points[i, 1]*1e3, 0, bar_width*1e3, bar_width*1e3, z[i], color=color, **kwargs)
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_zlabel('Area fraction')

        # Plot ConvexHull objects on the xy plane
        for index in range(len(self.polygons)):
            self.plot_polygon(index, ax=ax, fill=color, alpha_fill=0.2, include_area_value=True) 
        
    
    def plot_polygon(self, index, ax=None, fill=None, show_sample=False, alpha_fill=1.0, include_area_value=False, marker_kwargs={}, **kwargs):
        # Get the polygon
        polygon = self.polygons[index]

        # Create a figure and axes
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlim(self.xbounds)
            ax.set_ylim(self.ybounds)
        else:
            fig = ax.get_figure()

        # For each simplex in the polygon, draw the corresponding triangle

        my_kwargs = {"color":"k", "linewidth":2, "alpha":0.6}
        my_kwargs.update(kwargs)    
        for simplex in polygon.simplices:
            ax.plot(polygon.points[simplex, 0], polygon.points[simplex, 1], **my_kwargs)

        # If a fill color was specified, fill in the polygon
        polygon_vertices = polygon.points[polygon.vertices]
        if fill is not None:
            polygon_patch = patches.Polygon(polygon_vertices, fill=True, color=fill, alpha=alpha_fill)
            ax.add_patch(polygon_patch)
        
        if include_area_value:
            centroid = np.mean(polygon_vertices, axis=0)
            # Annotate the plot with the area of the polygon at the centroid
            ax.text(centroid[0], centroid[1], f'Area: {polygon.volume:.2f}', horizontalalignment='center')

        if show_sample:
            my_kwargs = {"s": 50, "marker":"o", "edgecolors": "k", "linewidths": 1.5, "c": "b"}
            my_kwargs.update(marker_kwargs)
            ax.scatter(self.points[index, 0], self.points[index, 1],**my_kwargs)

def polygon_skewness(hull):
    # Extract the vertices of the polygon
    vertices = hull.points[hull.vertices]
    
    # Compute the center of mass of the polygon
    center_of_mass = np.mean(vertices, axis=0)
    
    # Compute distances from the center of mass to each vertex
    distances = cdist([center_of_mass], vertices)[0]
    
    # Compute skewness as maximum distance divided by minimum distance
    skewness = np.max(distances) / np.min(distances)
    
    return skewness

def ticker_meters_to_mm(x, pos):
    'Converts meters to millimeters'
    return "{:.1f}".format(x*1000)

