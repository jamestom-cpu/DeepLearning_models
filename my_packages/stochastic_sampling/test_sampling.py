import numpy as np


from scipy import stats, spatial
from matplotlib import pyplot as plt
from matplotlib import ticker as tkr


import pyDOE2 as doe
from shapely.geometry import Polygon



from my_packages.stochastic_sampling.voronoi import VoronoiAnalysis
from my_packages.classes.field_classes import Scan



class Stochastic_Sampling():
    def __init__(
            self, 
            F: Scan,
            n_points: int,
            seed= None,
            ):
        self.F = F
        self.xbounds = np.asarray(self.F.grid.bounds()[0])
        self.ybounds = np.asarray(self.F.grid.bounds()[1])
        self.n_points = n_points
        self.points = None
        self.measurements = None
        self.tri = None
        self.vor = None
        self.hull = None
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    

        
    def sample_MC(self):
        x = [np.random.uniform(
            low=self.F.grid.bounds()[0][0], 
            high=self.F.grid.bounds()[0][1], 
            size = self.n_points
            )]
        y = [np.random.uniform(
            low=self.F.grid.bounds()[1][0], 
            high=self.F.grid.bounds()[1][1],
            size = self.n_points 
            )]
        self.points = np.vstack((x,y)).T
        return self
    
    def _rescale_points(self, points: np.ndarray):
        x = points[:,0]*(self.F.grid.bounds()[0][1] - self.F.grid.bounds()[0][0]) + self.F.grid.bounds()[0][0]
        y = points[:,1]*(self.F.grid.bounds()[1][1] - self.F.grid.bounds()[1][0]) + self.F.grid.bounds()[1][0]
        return np.vstack((x,y)).T
    
    def sample_halton(self):
        halton_sampler = stats.qmc.Halton(2, scramble=True, seed=self.seed)
        self.points = self._rescale_points(halton_sampler.random(self.n_points))  
        return self
    
    def sample_sobol(self):
        sobol_sampler = stats.qmc.Sobol(2, scramble=True, seed=self.seed)
        self.points = sobol_sampler.random(self.n_points)
        self.points = self._rescale_points(self.points)
        return self
    
    def sample_latin(self):
        latin_sampler = stats.qmc.LatinHypercube(2, scramble=True, seed=self.seed)
        self.points = latin_sampler.random(self.n_points)
        self.points = self._rescale_points(self.points)
        return self
    def sample_latin_doe(self, **kwargs):
        """
        
        Parameters
        ----------
        criterion: str  =>  Allowable values are 
         * "center" or "c", 
         * "maximin" or "m", "centermaximin" or "cm"
         * "correlation" or "corr"
        . If no value given, the design is simply randomized.

        iterations: int  =>  Number of iterations in the maximin and correlations algorithms (default 5).

        """
        my_kwargs = {"criterion":"centermaximin",
                     "iterations": 10,
                     "random_state": self.seed
                     }
        my_kwargs.update(kwargs)

        print(my_kwargs)
        samples = doe.lhs(2, samples=self.n_points, **my_kwargs)    
        self.points = self._rescale_points(samples)
        return self
    
    def measure_points(self):
        self.measurements = np.array([self.F.sample_point(p) for p in self.points])
        return self
    
    def _compute_delaunay(self):
        return spatial.Delaunay(self.points)
    
    def _compute_voronoi(self, scale_to_mm = False):
        return spatial.Voronoi(self.points if not scale_to_mm else self.points*1e3)
    
    def run_voronoi_analysis(self) -> VoronoiAnalysis:
        voronoi_analysis = VoronoiAnalysis(
            self.points, self.F.grid.bounds()[0], 
            self.F.grid.bounds()[1]
            )
        return voronoi_analysis     
    
    
    def _plot_scatter(self, units="V/m", ax=None, title="Sampling", **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else: 
            fig = ax.get_figure()
        
        my_kwargs = {"cmap":'viridis', "edgecolors":'none'}
        my_kwargs.update(kwargs)
        q = ax.scatter(self.points[:,0]*1e3, self.points[:,1]*1e3, c=self.measurements, **my_kwargs)
        # Add a colorbar for intensity scale
        cbar = fig.colorbar(q, ax=ax, label=units)
        ax.set_title(title)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        return ax, q
    
    def _plot_voronoi(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        # Plot
        my_kwargs = {"show_vertices":False, "line_colors":'k', "show_points":False, "line_width":2, "line_alpha":0.6}
        my_kwargs.update(kwargs) 
        self.vor = self._compute_voronoi(scale_to_mm=True)

        q = spatial.voronoi_plot_2d(self.vor, ax=ax, **my_kwargs)
        return ax, q
    
    def plot_measurements_voronoi(self, ax=None, units="V/m", title="Measurements", **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_aspect('equal')
            ax.set_xlim(self.xbounds*1.1)
            ax.set_ylim(self.ybounds*1.1)
            formatting_func = tkr.FuncFormatter(lambda x, pos: f"{x*1e3:.1f}")
            ax.xaxis.set_major_formatter(formatting_func)
            ax.yaxis.set_major_formatter(formatting_func)
            
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")

        self.measure_points()

        measurements = self.measurements
        vor_obj = self.run_voronoi_analysis()
        vor_obj.plot_all_polygons(show_samples=False, ax=ax)
        q = ax.scatter(*self.points.T, c=measurements, s=100, alpha=1)
        cbar = plt.colorbar(q, ax=ax, label=units)
        ax.set_title(title)
        return ax, q, cbar
    
    def plot_voronoi_statistics(self, title="", **kwargs):
        return plot_voronoi_statistics(self, title=title, **kwargs)
        
    def plot(self, ax=None, title="Sampling", **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.get_figure()
        ax, q = self._plot_scatter(ax=ax, title = title, **kwargs)
        self._plot_voronoi(ax=ax)

        # Set plot limits
        xbounds = np.asarray(self.F.grid.bounds()[0])*1e3
        ybounds = np.asarray(self.F.grid.bounds()[1])*1e3
        ax.set_xlim(xbounds)
        ax.set_ylim(ybounds)
        return ax, q



def plot_voronoi_statistics(ss: Stochastic_Sampling, include_model_fitting=True, title=None, **kwargs):
    fig, all_ax = plt.subplots(4, 2, figsize=(10, 8))
    ax1, ax2, ax3, ax4 = all_ax
    if title is None:
        my_title = f"Voronoi statistics for {ss.n_points} points"
    else:
        my_title = " - ".join([title, f"Voronoi statistics for {ss.n_points} points"])

    fig.subplots_adjust(hspace=1.0)
    n_subplots = 4
    # Add titles to each row

    titles = ["Monte Carlo", "Latin Hypercube", "Halton", "Sobol"]
    for i in range(n_subplots):
        # Create a new axes instance above the subplots in the current row
        ax = fig.add_subplot(n_subplots, 1, i + 1)
        ax.axis('off')  # Hide the axes

        # Add a title to the new axes instance, which will be displayed above the subplots
        ax.set_title(titles[i], y=1.1, fontsize=16)

    vorSt = ss.sample_MC().run_voronoi_analysis()
    vorSt.plot_histograms(ax=ax1, include_model_fitting=include_model_fitting, **kwargs)

    vorSt = ss.sample_latin().run_voronoi_analysis()
    vorSt.plot_histograms(ax=ax2, include_model_fitting=include_model_fitting, **kwargs)

    vorSt = ss.sample_halton().run_voronoi_analysis()
    vorSt.plot_histograms(ax=ax3, include_model_fitting=include_model_fitting, **kwargs)

    vorSt = ss.sample_sobol().run_voronoi_analysis()
    vorSt.plot_histograms(ax=ax4, include_model_fitting=include_model_fitting, **kwargs)

    fig.subplots_adjust(hspace=0.7)

    for axx in all_ax:
        axx[0].set_xlim(0, 0.035)
        axx[1].set_xlim(1, 4)

    fig.suptitle(my_title, fontsize=16, y=0.97)
    return fig, all_ax