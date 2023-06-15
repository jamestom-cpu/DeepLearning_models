## Sequential Sample Generators
from collections import namedtuple

from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import numpy as np
import sobol_seq


from .spatial_analysis import SpatialAnalysis, VoronoiAnalysis, KDTree_Analysis
from .sample_simulator import SampleSimulator
from .helper_functions.sequential_sampling_funcs import *
from .helper_functions.voronoi_funcs import convexhull_to_polygon

DataSample = namedtuple('DataSample', ['point', 'measurement'])


class SequentialSampler():
    def __init__(
            self, 
            sample_simulator: SampleSimulator,
            xbounds=None, 
            ybounds=None,
            seed=42,
            include_default_voronoi: bool = True
            ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.sampler = sample_simulator    
        self.xbounds = np.asarray(sample_simulator.xbounds) if xbounds is None else np.asarray(xbounds)
        self.ybounds = np.asarray(sample_simulator.ybounds) if ybounds is None else np.asarray(ybounds)
        self.start_sobol_sampler()
        self.data_samples = []
        self.probe_height = self.sampler.zbounds[0]
        self.default_leafsize = 3
        self.include_default_voronoi = include_default_voronoi  
    
    @property
    def points(self):
        return np.asarray([dp.point for dp in self.data_samples])
    
    @property
    def measurements(self):
        return np.asarray([dp.measurement for dp in self.data_samples])
    
    def start_sobol_sampler(self, init_index = 1):
        self.sobol_sampler = self._sobol_generator(init_index)
        return self
    
    def _sobol_generator(self, init_index=1):
        i = init_index
        while True:
            # Generate the next point in the Sobol sequence
            point = sobol_seq.i4_sobol(2, i)[0]
            # Rescale to the given bounds
            x = self.xbounds[0] + point[0] * (self.xbounds[1] - self.xbounds[0])
            y = self.ybounds[0] + point[1] * (self.ybounds[1] - self.ybounds[0])
            yield (x, y)
            i += 1

    def sample_corners(self, buffer_factor=0.01):
        xbounds = self.xbounds*(1-buffer_factor)
        ybounds = self.ybounds*(1-buffer_factor)
        corner_points = np.asarray([
            [xbounds[0], ybounds[0], self.probe_height],
            [xbounds[0], ybounds[1], self.probe_height],
            [xbounds[1], ybounds[0], self.probe_height],
            [xbounds[1], ybounds[1], self.probe_height],
            ])
        for point in corner_points:
            self.measure_point(point)
        return self


    def sobol_sample(self):
        point = np.asarray(next(self.sobol_sampler)+(self.probe_height,))
        measurement = self.sampler.interpolated_sample(point)
        data_point = DataSample(point, measurement)
        self.data_samples.append(data_point)
        return point

    def sobol_sample_n_points(self, n_points):
        return np.asarray([self.sobol_sample() for _ in range(n_points)])

    
    def measurements_on_groups_kdt(self, group_dict=None):
        if group_dict is None:
            group_dict = self.Spatial.KDTree().leaf_dict
        return {key: self.measurements[group_dict[key]] for key in group_dict}
    
    def error_func_kdt(self, point_index: int , group_dict: dict, group_index: int):
        assert point_index in group_dict[group_index], "point_index must be in group_index"
        # point_m = self.measurements[point_index]
        measurements = [self.measurements[point] for point in group_dict[group_index] if point != point_index]
        return np.std(measurements)
        # mean_value = np.mean(other_points)
        # if other_points == []:
        #     return 0
        # if mean_value == 0:
        #     return np.abs(point_m)
        # return np.abs(point_m - mean_value)/mean_value

    def kdt_leaf_error(self, group_index, kdtree=None):
        if kdtree is None:
            kdtree = self.Spatial.KDTree()   
        group_dict = kdtree.leaf_dict
        leaf_areas = kdtree.leaf_areas
        point_errors = np.asarray([self.error_func_kdt(point_index, group_dict, group_index) for point_index in group_dict[group_index]])
        leaf_error = np.mean(point_errors)*leaf_areas[group_index]
        return leaf_error


    def calculate_kdt_leaf_errors(self, leafsize=None, kdtree: KDTree_Analysis=None):
        if leafsize is None:
            leafsize = self.default_leafsize
        if kdtree is None:
            kdtree = self.Spatial.KDTree(leafsize)
        self.errors = np.asarray([self.kdt_leaf_error(group_index, kdtree) for group_index in kdtree.leaf_dict.keys()])
        return self
    
    def _sample_extra_N_points_kdt(self, leaf_index, N, kdtree: KDTree_Analysis):
        xmin, xmax, ymin, ymax = kdtree.rectangles[leaf_index]
        circle_center = ((xmin+xmax)/2, (ymin+ymax)/2)
        circle_radius = np.sqrt((xmin-circle_center[0])**2 + (ymin-circle_center[1])**2)
        points = sample_points_within_circle(circle_center, circle_radius, N, self.xbounds, self.ybounds)
        return points

    def sequential_step_kdt(self, leafsize=None, N_points=10):
        if leafsize is None:
            leafsize = self.default_leafsize
        kdtree = self.Spatial.KDTree()
        self.calculate_kdt_leaf_errors(leafsize=leafsize, kdtree=kdtree)
        leaf_index = np.argmax(self.errors)
        
        new_points = self._sample_extra_N_points_kdt(leaf_index, N_points, kdtree)
        leaf_points = kdtree.points[kdtree.leaf_dict[leaf_index]]
        point = find_sample_with_largest_distance(new_points, leaf_points)
        point = np.append(point, self.probe_height)
        self.measure_point(point)
        return self
    
    def return_area_metric(self, vor: VoronoiAnalysis=None):
        if vor is None:
            vor = self.Spatial.Vor
        return vor.norm_areas


    def _return_gradient_residual(
            self, vor: VoronoiAnalysis, index: int, 
            k_nearest = -1, regularization_term=1e-3, normalized=False):
        
        if k_nearest == -1:
            neighbor_indices = vor.find_adjacent_neighbors(index)["indices"]
        else:
            neighbor_indices = vor.find_k_nearest_neighbors(index, k_nearest)["indices"]

        neighbor_points = vor.points[neighbor_indices]
        my_point = vor.points[index]
        measured_neighbors = self.measurements[neighbor_indices]
        measured_point = self.measurements[index]

        delta_r = neighbor_points - my_point 
        delta_F = measured_neighbors - measured_point
        gradient_best_fitted, residual = find_optimum_matrix(
            delta_r, delta_F, alpha=regularization_term, normalized=normalized)
        return gradient_best_fitted, residual

    def cell_gradient_residuals(self, vor: VoronoiAnalysis=None, normalize=True,
        k_nearest = -1, regularization_term=1e-3, single_cell_normalized=False):
        if vor is None:
            vor = self.Spatial.Vor
        residuals = []
        for index in range(len(vor.points)):
            self.gradients, residual = self._return_gradient_residual(
                vor, index, k_nearest=k_nearest, regularization_term=regularization_term, normalized=single_cell_normalized)
            residuals.append(residual)
        
        if normalize:
            residuals = np.asarray(residuals) / np.sum(residuals)
        self.residuals = residuals
        return residuals
    
    def sequential_step_gradient(self, k_nearest_neighbors=4, n_random_points=50, regularization_term=1e-4, buffer=-1, increase_k_factor=1.5):
        vor = self.Spatial.Vor
        gradient_residuals = self.cell_gradient_residuals(
            vor, normalize=True, k_nearest=k_nearest_neighbors, 
            single_cell_normalized=False, regularization_term=regularization_term)
        area_metric = self.return_area_metric(vor)
        total_metric = gradient_residuals + area_metric
        largest_error_index = np.argmax(total_metric)
        if buffer == -1:
            sampling_region = convexhull_to_polygon(vor.polygons[largest_error_index])
        else:
            sampling_region = self._determine_sampling_region_with_neighboring_samples(vor, largest_error_index, k_nearest_neighbors, buffer=buffer)
        candidate_points = sample_N_points_from_polygon(sampling_region, n_random_points)
        fixed_points = [vor.points[largest_error_index]] \
            + vor.points[vor.find_k_nearest_neighbors(
            largest_error_index, int(increase_k_factor*k_nearest_neighbors))["indices"]]
        point = find_sample_with_largest_distance(candidate_points, fixed_points)
        self.measure_point(point)
        return self
    
    
    def measure_point(self, point):
        if len(point)==2:
            point = np.append(point, self.probe_height)
        measurement = self.sampler.interpolated_sample(point)
        data_point = DataSample(point, measurement)
        self.data_samples.append(data_point)
        return self
    

    @property
    def Spatial(self):
        points = self.points[:, :2]
        self.spatial_analysis = SpatialAnalysis(
            points, self.xbounds, self.ybounds, seed=self.seed, 
            default_leafsize=self.default_leafsize, include_default_voronoi=self.include_default_voronoi
            )
        return self.spatial_analysis
    
    @staticmethod
    def _determine_sampling_region_union_of_cells(vor: VoronoiAnalysis, largest_error_index: int, k_neighbors: int)-> Polygon:
        # get the polygon of the cell
        polygon = convexhull_to_polygon(vor.polygons[largest_error_index])
        # get the nearest neighbors
        nearest_neighbors_indices = vor.find_k_nearest_neighbors(largest_error_index, k_neighbors)["indices"]
        nearest_neighbors = [convexhull_to_polygon(vor.polygons[ii]) for ii in nearest_neighbors_indices]
        # create a sampling region
        sampling_region = unary_union([polygon, *nearest_neighbors])
        return sampling_region
    
    @staticmethod
    def _determine_sampling_region_with_neighboring_samples(vor: VoronoiAnalysis, largest_error_index: int, k_neighbors: int, buffer: int=0)-> Polygon:
        # get the polygon of the cell
        polygon = convexhull_to_polygon(vor.polygons[largest_error_index])
        # get the nearest neighbors
        nearest_neighbors_indices = vor.find_k_nearest_neighbors(largest_error_index, k_neighbors)["indices"]
        # get voronoi cell points
        cell_points = [Point(x, y) for x, y in polygon.exterior.coords]
        # get the samples from the nearest neighbors
        new_points = [Point(p) for p in vor.points[nearest_neighbors_indices]]
        # create a sampling region
        sampling_region = Polygon(cell_points + new_points).convex_hull
        # normalize the buffer 
        xmin, ymin, xmax, ymax = sampling_region.bounds
        average_axis = np.mean([xmax-xmin, ymax-ymin])
        buffer = buffer*average_axis
        return sampling_region.buffer(buffer)