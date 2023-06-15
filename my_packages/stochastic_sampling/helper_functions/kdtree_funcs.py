import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.ckdtree import cKDTreeNode
import matplotlib.pyplot as plt


def get_rectangles_and_indices(node: cKDTreeNode, bounds):
    rectangles = []
    indices = []

    if hasattr(node, "_node"):
        node = node._node

    if node.lesser is None and node.greater is None:
        # We are at a leaf node. Store the rectangle and indices.
        rectangles.append(bounds)
        indices.append(node.indices)
    else:
        # We are at an inner node. Get the partitions of the children.
        dim = node.split_dim
        split = node.split

        new_bounds = [
            [bounds[0], split, bounds[2], bounds[3]],
            [split, bounds[1], bounds[2], bounds[3]],
            [bounds[0], bounds[1], bounds[2], split],
            [bounds[0], bounds[1], split, bounds[3]]
        ]
        
        for child, i in zip([node.lesser, node.greater], [0, 1]):
            if child is not None:
                rects, inds = get_rectangles_and_indices(child, new_bounds[i+2*(dim==1)])
                rectangles.extend(rects)
                indices.extend(inds)

    return rectangles, indices


def get_kdtree_partitions(points, leafsize, bounds=None):
    # Create the KDTree.
    tree = KDTree(points, leafsize=leafsize)

    if bounds is None:
        # Compute the bounds of the points.
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
    else:
        (x_min, x_max), (y_min, y_max) = bounds

    # Get the KDTree partitions and indices.
    rectangles, indices = get_rectangles_and_indices(tree.tree, [x_min, x_max, y_min, y_max])

    return rectangles, indices
