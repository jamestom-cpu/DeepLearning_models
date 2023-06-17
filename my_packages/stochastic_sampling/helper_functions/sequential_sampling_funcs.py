import numpy as np
from scipy.stats import qmc
import random
from shapely.geometry import Polygon, Point



def sample_points_within_circle(circle_center, radius, N, xbounds=None, ybounds=None):
    points = []
    center_x, center_y = circle_center
    
    while len(points) < N:
        # Generate random angle
        angle = random.uniform(0, 2 * np.pi)

        # Calculate random radius within the circle
        random_radius = np.sqrt(random.uniform(0, 1)) * radius

        # Calculate coordinates of the point
        x = center_x + random_radius * np.cos(angle)
        y = center_y + random_radius * np.sin(angle)

        if xbounds is not None and ybounds is not None:
            if x < xbounds[0] or x > xbounds[1] or y < ybounds[0] or y > ybounds[1]:
                continue
        points.append([x, y])
    return np.asarray(points)


def sample_points_within_circle_Sobol(circle_center, radius, N, xbounds=None, ybounds=None):
    points = []
    points = qmc.Sobol(2).random(N)
    points = (points - 0.5)*2*radius + np.asarray(circle_center)
    # filter points outside of circle
    points = points[np.linalg.norm(points, axis=1) <= radius]
    if xbounds is not None and ybounds is not None:
        points = points[np.logical_and(points[:, 0] >= xbounds[0], points[:, 0] <= xbounds[1])]
        points = points[np.logical_and(points[:, 1] >= ybounds[0], points[:, 1] <= ybounds[1])]
    n_missing_points = N - len(points)
    if n_missing_points > 0:
        points = np.concatenate([points, sample_points_within_circle(circle_center, radius, n_missing_points, xbounds, ybounds)])
    return points

def sample_N_points_from_polygon(polygon: Polygon, N):
    points = []
    while len(points) < N:
        x = np.random.uniform(polygon.bounds[0], polygon.bounds[2], 1)
        y = np.random.uniform(polygon.bounds[1], polygon.bounds[3], 1)
        point = (x, y)
        if polygon.contains(Point(point)):
            points.append(point)
    return np.asarray(points).squeeze()




def find_sample_with_largest_distance(new_points, fixed_points):
        max_distance = -1
        selected_sample = None

        for sample_x in new_points:
            distances = np.linalg.norm(sample_x - fixed_points, axis=1)  # Calculate Euclidean distances
            max_distance_for_sample_x = np.max(distances)

            if max_distance_for_sample_x > max_distance:
                max_distance = max_distance_for_sample_x
                selected_sample = sample_x

        return selected_sample

def find_optimum_matrix(delta_x, delta_f, alpha, rcond=None, normalized = False):
    # normalize delta_f 
    max_delta_f = np.max(delta_f)
    delta_f = delta_f / max_delta_f 

    m, n = delta_x.shape
    regularization_term = alpha * np.eye(n)
    g, _, _, _ = np.linalg.lstsq(
        delta_x.T.dot(delta_x) + regularization_term, delta_x.T.dot(delta_f), rcond=rcond) 
    
    if normalized:
        residual = np.sum(np.abs(delta_f - delta_x.dot(g)))
        return g.T, residual
    else:
        g =  g * max_delta_f
        residual = np.sum(np.abs(delta_f - delta_x.dot(g)))
        return g.T, residual

