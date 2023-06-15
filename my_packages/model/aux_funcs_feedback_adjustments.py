import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from skimage.transform import resize
from skimage.feature import peak_local_max

from my_packages.classes.dipole_array import FlatDipoleArray

def perform_clustering(data, num_clusters: int):
    # Convert the data to a numpy array
    data_array = np.array(data)

    # Create a mask for NaN values
    nan_mask = np.isnan(data_array)

    # Replace NaN values with a placeholder value
    data_array[nan_mask] = -1

    # Reshape the data to a 2D array
    X = data_array.reshape(-1, 1)

    # Perform clustering with K-means
    my_kwargs = {'n_clusters': num_clusters} 
    kmeans = KMeans(**my_kwargs)
    kmeans.fit(X)

    # Get the cluster labels
    labels = kmeans.labels_

    # Determine the cluster with the lowest values
    lowest_cluster = np.argmin(kmeans.cluster_centers_)

    # Assign labels '0' and '1' based on the lowest cluster
    labels = np.where(labels == lowest_cluster, 0, 1).reshape(data_array.shape)

    return labels

def return_strongest_dipoles(dipole_array: FlatDipoleArray, num_clusters: int) -> FlatDipoleArray:
    class_labels = perform_clustering(np.abs(dipole_array.M.flatten()), num_clusters)
    sil_score = silhouette_score(np.abs(dipole_array.M.flatten()).reshape(-1,1), class_labels)
    # if sil_score < 0.7:
    #     print("Silhouette score is too low")
    #     return dipole_array
    new_dipole_list = np.asarray(dipole_array.dipoles)[class_labels==1]
    new_dipole_array = FlatDipoleArray.init_dipole_array_from_dipole_list(f = dipole_array.f, dipoles = new_dipole_list)
    new_dipole_array.height = dipole_array.height
    return new_dipole_array


def conservative_peak_finder(data: np.ndarray, conservativeness: float = 0.5):
    # Input validation
    assert data.ndim == 2, "Input data must be a 2D array"
    assert isinstance(conservativeness, (int, float)) and 0 <= conservativeness <= 1, \
        "conservativeness must be a number between 0 and 1"

    # Calculate threshold: mean + conservativeness * standard deviation
    threshold_abs = np.mean(data) + conservativeness * 5 * np.std(data)

    # Identify peaks
    peak_coords = peak_local_max(data, min_distance=1, threshold_abs=threshold_abs)

    return peak_coords

def downsample_error_map(a: np.ndarray, target_shape: tuple):
    # Assert that target_shape is indeed a tuple with 2 elements
    assert isinstance(target_shape, tuple) and len(target_shape) == 2, "target_shape must be a tuple with 2 elements"
    
    P, Q = target_shape

    # Assert that P and Q are valid
    assert P > 0 and Q > 0, "P and Q must be positive integers"

    # Calculate the new shape
    new_shape = (P, Q)

    # Downsample using the area-based method
    b = resize(a, new_shape, mode='reflect', anti_aliasing=False)

    # Multiply each cell in the downsampled image by the number of cells it represents to get the cumulative error
    b *= (a.shape[0] / P) * (a.shape[1] / Q)
    
    return b