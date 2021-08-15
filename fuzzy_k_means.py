# python -m unittest fuzzy_k_means

import numpy as np
from distance_metrics import euclidean_distance


def normalise(memberships):
    cumulative = np.sum(memberships, axis=0, keepdims=True)
    return memberships / cumulative


def get_cluster_centres(dataset, memberships, b):
    number_of_clusters, _ = memberships.shape
    centroids = []
    for centroid_i in range(number_of_clusters):
        memberships_exponent = np.power(memberships[centroid_i, :], b)
        numerator = np.sum(memberships_exponent * dataset, axis=1)
        denominator = np.sum(memberships_exponent)
        centroids.append(numerator / denominator)
    return centroids


def get_memberships(dataset, centroids, b):
    _, number_of_samples = dataset.shape
    number_of_clusters = len(centroids)
    memberships = np.zeros((number_of_clusters, number_of_samples))
    for sample_i in range(number_of_samples):
        sample = dataset[:, sample_i]
        for centroid_i, centroid in enumerate(centroids):
            distance = euclidean_distance(sample, centroid)
            memberships[centroid_i, sample_i] = np.power(1/distance, 2/(b-1))
    return normalise(memberships)


S = np.array([
    [-1, 1, 0, 4, 3, 5],
    [3, 4, 5, -1, 0, 1]
])

mu = np.array([
    [7, 4, 0.5, 0.5, 0.5, 0],
    [0, 4, 0.5, 0.5, 0.5, 1]
])

cluster_centres = get_cluster_centres(S, normalise(mu), 2)
print(get_memberships(S, cluster_centres, 2))



