import numpy as np
from scipy.spatial import Voronoi, ConvexHull
from itertools import product
from shapely.geometry import Polygon, GeometryCollection, Point, LineString, MultiPolygon, MultiLineString
from shapely.ops import unary_union
from typing import Tuple, Iterable





def return_large_polygon(vor, index, xbounds, ybounds):
    ridges = find_ridge_vertices(vor, index)
    open_ridge_indices = return_open_ridge_indices(ridges)
    open_r = return_kernels_and_side_samples_open_cell(vor, index, verbose=False)

    A1, A2 = [return_finite_point_of_open_ridge(ridges[i]) for i in open_ridge_indices]
    m1,m2 = get_slopes(open_r)
    
    x_ptp, y_ptp = np.ptp(vor.vertices, axis=0)
    scale = np.ptp(xbounds) + np.ptp(ybounds) + x_ptp + y_ptp

    P1 = translate_along_line(Point(A1), m1, scale)
    P2 = translate_along_line(Point(A2), m2, scale)
    P3 = translate_along_line(Point(A1), m1, -scale)
    P4 = translate_along_line(Point(A2), m2, -scale)

    

    # check if there are intersections
    l1 = LineString([Point(P1), Point(P3)]); l2 = LineString([Point(P2), Point(P4)])
    intersection_point = l1.intersection(l2)
    
    if intersection_point.is_empty:
        return Polygon([Point(P1), Point(P2), Point(P4), Point(P3)]).convex_hull
    
    else:
        assert isinstance(intersection_point, Point), "the intersection point must be a point, not {}".format(type(intersection_point))
        poly1 = Polygon([Point(P1), Point(P2), Point(intersection_point)])
        poly2 = Polygon([Point(P3), Point(P4), Point(intersection_point)])
        poly3 = Polygon([Point(P1), Point(P4), Point(intersection_point)])
        poly4 = Polygon([Point(P2), Point(P3), Point(intersection_point)])

        sample_point = Point(vor.points[index])

        for p in [poly1, poly2, poly3, poly4]:
            if p.contains(sample_point):
                return p
        else:
            raise ValueError("the sample point is not contained in the polygon")


def create_boundbox(xbounds, ybounds):
    rect_vertices = [Point(v) for v in product(xbounds, ybounds)]
    rect = unary_union(rect_vertices).convex_hull
    return rect

def extract_polygons(geometry_collection):
    return [geom for geom in geometry_collection.geoms if isinstance(geom, Polygon)]


def remove_overlap(
        index: int, 
        vor: Voronoi,  
        polygon_list: Iterable[Polygon],
        xbounds:Tuple[float,float], 
        ybounds:Tuple[float,float], 
        tol=1e-10):
    my_poly = polygon_list[index]
    adjacent_cells = find_voronoi_cell_adjacent_neighbors(vor, index, xbounds, ybounds)
    adjacent_polygons = [polygon_list[ii] for ii in adjacent_cells]

    def filter_func(overlap):
        if isinstance(overlap, (LineString, MultiLineString)):
            return False
        elif isinstance(overlap, Polygon):
            if overlap.is_empty:
                return False
            if overlap.area < tol:
                return False
            return True

    overlapping_polygons = []
    overlapping_indices = []
    # check if there is any overlap between my_poly and adjacent_polygons
    for ii, p in zip(adjacent_cells, adjacent_polygons):
        if my_poly.intersects(p) and filter_func(my_poly.intersection(p)):
            overlapping_polygons.append(p)
            overlapping_indices.append(ii)
    
    if len(overlapping_polygons) == 0:
        return my_poly
    
    # get the shared ridges
    my_ridges = find_ridges(vor, index)
    overlapping_ridges = []
    for ind in overlapping_indices:
        overlapping_ridges.extend(find_ridges(vor, ind))

    # common_ridges excluding the -1 index
    common_ridges = [r for r in my_ridges if r in overlapping_ridges and -1 not in r]
    ridge_points = [vor.vertices[r] for r in common_ridges]
    separation_line = MultiLineString([LineString(r) for r in ridge_points])

    # get union of overlapping polygons
    overlapping_polygons = unary_union(overlapping_polygons+[my_poly])
    diff = overlapping_polygons.difference(separation_line.buffer(tol))

    if not isinstance(diff, (GeometryCollection, MultiPolygon)):
        print("Warning: GeometryCollection not returned")
        return diff
    sample = vor.points[index]
    my_poly = diff.geoms[0] if diff.geoms[0].contains(Point(sample)) else diff.geoms[1]
    return my_poly.union(separation_line).convex_hull


def return_side_polygons(vor, index, union_polygon, xbounds, ybounds)->Polygon:  
    boundbox = create_boundbox(xbounds, ybounds)
    large_polygon = return_large_polygon(vor, index, xbounds, ybounds)
    poly1 = large_polygon.difference(union_polygon)
    geom_coll = boundbox.intersection(poly1)
    if isinstance(geom_coll, (GeometryCollection, MultiPolygon)):
        poly_list = extract_polygons(geom_coll)
        for p in poly_list:
            if p.contains(Point(vor.points[index])):
                return p
        raise ValueError("the polygon does not contain the point")

    elif isinstance(geom_coll, Polygon):
        return geom_coll
    elif geom_coll.is_empty:
        raise ValueError("the geometry collection is empty")
    else:
        raise TypeError("the geometry collection must be either a polygon or a geometry collection, not {}".format(type(geom_coll)))
   
def shapely_polygons_to_convex_hull(polygon: Iterable[Polygon]):
    return [ConvexHull(p.exterior.coords) for p in polygon]  

def get_polygons(vor, xbounds, ybounds):
    internal_polygons = closed_cells_to_polygons(vor, xbounds, ybounds, verbose=False)
    internal_polygons = [convexhull_to_polygon(p) for p in internal_polygons]
    union_polygon  = unary_union(internal_polygons)
    
    open_cell_indices = [i for i in range(len(vor.points)) if not is_cells_closed(vor, i)]
    closed_cell_indices = [i for i in range(len(vor.points)) if is_cells_closed(vor, i)]

    all_polygons = np.zeros(len(vor.points), dtype=object)
    all_polygons[closed_cell_indices] = internal_polygons

    side_polygons = [v for v in [return_side_polygons(vor, ii, union_polygon, xbounds, ybounds) for ii in open_cell_indices] if not v.is_empty] 
    all_polygons[open_cell_indices] = side_polygons


    for ii in open_cell_indices:
        all_polygons[ii] = remove_overlap(ii, vor, all_polygons, xbounds, ybounds, tol=1e-10)
    
    all_polygons = shapely_polygons_to_convex_hull(all_polygons)
    return all_polygons

def get_slopes(open_r: Tuple[Point, Tuple[Point, Point]]):
    assert len(open_r) == 2, "the number of open ridges must be 2, not {}".format(len(open_r))
    m = []

    for k, (p1, p2) in open_r: 
        half_point = (np.array([p1.x, p1.y]) + np.array([p2.x, p2.y])) / 2
        k = np.array([k.x, k.y])
        
        if half_point[0] == k[0]:  # vertical line
            m.append(np.inf)
        else:  # non-vertical line
            m.append((k[1] - half_point[1]) / (k[0] - half_point[0]))
    
    return m

def translate_along_line(A: Point, m: float, r):
    r_sign = r/np.abs(r)

    # infinite m1
    if np.isinf(m):
        xB = A.x
        yB = A.y + r_sign * r
        B = Point(xB, yB)
        return B


    # intercept of the line
    b = A.y - m * A.x # from y = mx + b -> b = y - mx

    xB = A.x + r_sign*np.sqrt(r**2 / (1 + m**2)) # from x^2 + y^2 = r^2 -> x = sqrt(r^2 / (1 + m^2))
    yB = m * xB + b # from y = mx + b
    
    B = Point(xB, yB)
    return B



def closed_cells_to_polygons(vor: Voronoi, xbounds: Tuple, ybounds: Tuple, verbose=False):
    def mprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs) 

    rect_vertices = np.asarray(list(product(xbounds, ybounds)))
    rect = ConvexHull(rect_vertices)
    rpoly = convexhull_to_polygon(rect) 

    closed_polygons = [] 
    for i in range(len(vor.points)):
        # check if the cell is closed
        if not is_cells_closed(vor, i):
            mprint("cell {} is not closed".format(i), end="\r ")
            continue
        else:
            mprint("cell {} is closed".format(i), end="\r ")
            segm = return_closed_cell_segments(vor, i)
            verts = segments_to_vertices(segm)
            largeCell = ConvexHull(verts)
            polygon_cell = convexhull_to_polygon(largeCell)
            smallPolygonCell = rpoly.intersection(polygon_cell)
            # check if the cell is empty
            if smallPolygonCell.is_empty:
                mprint("cell {} is empty".format(i), end="\r ")
                continue
            else:
                mprint("cell {} is not empty".format(i), end="\r ")
                polygon = ConvexHull(smallPolygonCell.exterior.coords)
                closed_polygons.append(polygon)
    return closed_polygons

def convexhull_to_polygon(convex_hull):
    # Get the convex hull vertices in the correct order
    ordered_vertices = convex_hull.points[convex_hull.vertices]

    # Create a Polygon object from the ordered vertices
    polygon = Polygon(ordered_vertices)

    return polygon

def return_closed_cell_segments(vor, index):
    # assert the cell is closed
    assert(is_cells_closed(vor, index)), "the cell is not closed"
    # find the ridges of the cell
    ridges = find_ridge_vertices(vor, index)
    return ridges

def is_cells_closed(vor, index):
    ridges = find_ridge_vertices(vor, index)
    for ridge in ridges:
        if any(isinstance(v, (float, int)) and v==-1 for v in ridge):
            return False
    return True


def return_kernels_and_side_samples_open_cell(vor: Voronoi, index: int, verbose=False):
    def mprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    if is_cells_closed(vor, index):
        mprint("cell {} is closed".format(index), end="\r ")
        return []
    else:
        ridges = find_ridge_vertices(vor, index)
        open_ridge_indices = return_open_ridge_indices(ridges)
        open_ridge_points = np.array(find_ridge_points(vor, index))[open_ridge_indices]
        open_ridges_kernel = [return_finite_point_of_open_ridge(ridges[i]) for i in open_ridge_indices]
        list_of_obj = list(zip(open_ridges_kernel, open_ridge_points))
        return [[Point(k), (Point(p1), Point(p2))] for k, (p1, p2) in list_of_obj]
    
def find_ridge_points(vor, index):
    points = []
    for point_pair in vor.ridge_dict.keys():
        if index in point_pair:
            # Replace index values with corresponding point coordinates
            points.append(tuple(vor.points[i] for i in point_pair))
    return points

def return_finite_point_of_open_ridge(ridge: np.ndarray):
    inf_condition_on_point = lambda x : isinstance(x, (float, int)) and x == -1
    # get the vertex that is not -1
    v0, v1 = ridge
    finite_v = v0 if inf_condition_on_point(v1) else v1
    return finite_v 

def segments_to_vertices(closed_vertices: np.ndarray):
    closed_vertices = np.asarray(closed_vertices).reshape(-1, 2)
    return np.unique(closed_vertices, axis=0)




def return_open_ridge_indices(ridges: list): 
    return [i for i in range(len(ridges)) if is_ridge_open(ridges[i])]

def return_closed_ridge_indices(ridges: list): 
    return [i for i in range(len(ridges)) if not is_ridge_open(ridges[i])]


def is_ridge_open(ridge: np.ndarray):
    p1, p2 = ridge
    inf_condition_on_point = lambda x : isinstance(x, (float, int)) and x == -1
    return np.logical_or(inf_condition_on_point(p1), inf_condition_on_point(p2))

def find_ridges(vor, index):
    my_point_ridges = []
    for i, (point1, point2) in enumerate(vor.ridge_points):
        if index in [point1, point2]:
            my_point_ridges.append(vor.ridge_vertices[i])
    return my_point_ridges

def find_ridge_vertices(vor, index):
    ridge_indices = find_ridges(vor, index)
    # last_position = len(vor.vertices) - 1
    # ridge_indices[np.where(np.isin(ridge_indices, last_position))] = -1
    ridge_vertices = []
    for ridge in ridge_indices:
        vertices = [vor.vertices[i] if i != -1 else -1 for i in ridge]
        ridge_vertices.append(vertices)
    return ridge_vertices


def find_voronoi_cell_adjacent_neighbors(vor, index, xbounds, ybounds, return_shared_vertices=False):
    """Find the indices of neighboring points in a Voronoi diagram.

    Parameters
    ----------
    vor : scipy.spatial.Voronoi
        The Voronoi diagram.
    index : int
        The index of the point.
    xbounds : tuple of float
        The minimum and maximum x-coordinates (xmin, xmax).
    ybounds : tuple of float
        The minimum and maximum y-coordinates (ymin, ymax).

    Returns
    -------
    neighbors : list of int
        The indices of the neighboring points.
    """
    point_region_index = vor.point_region[index]  # The index of the region of the point
    point_region = vor.regions[point_region_index]  # The vertices of the region

    xmin, xmax = xbounds
    ymin, ymax = ybounds

    neighbors = []
    for i, region in enumerate(vor.regions):
        if i != point_region_index:
            common_vertices = [vertex for vertex in region if vertex in point_region]
            if common_vertices:  # If the regions share a vertex
                # Check if at least one common vertex is within the boundary box
                if any(xmin <= vor.vertices[vertex][0] <= xmax and ymin <= vor.vertices[vertex][1] <= ymax for vertex in common_vertices):
                    for point_idx, point_region_idx in enumerate(vor.point_region):
                        if point_region_idx == i:
                            neighbors.append(point_idx)
                            break

    return neighbors

def point_on_peryphery(point: Point, poly: Polygon, tol=1e-9):
    return poly.exterior.distance(point) < tol



# def find_voronoi_cell_adjacent_neighbors(vor, index):
#     """Find the indices of neighboring points in a Voronoi diagram.

#     Parameters
#     ----------
#     vor : scipy.spatial.Voronoi
#         The Voronoi diagram.
#     index : int
#         The index of the point.

#     Returns
#     -------
#     neighbors : list of int
#         The indices of the neighboring points.
#     """
#     point_region_index = vor.point_region[index]  # The index of the region of the point
#     point_region_vertices = vor.regions[point_region_index]  # The vertices of the region

#     neighbors = []
#     for i, region_vertices in enumerate(vor.regions):
#         if i != point_region_index and any(vertex in region_vertices for vertex in point_region_vertices):
            
#             for point_idx, point_region_idx in enumerate(vor.point_region):
#                 if point_region_idx == i:
#                     neighbors.append(point_idx)
#                     break

#     return neighbors