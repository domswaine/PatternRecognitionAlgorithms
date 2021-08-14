# python -m unittest k_means

import numpy as np
from distance_metrics import euclidean_distance

import unittest


def k_means(samples, centroids, metric):
    samples = [np.array(sample) for sample in samples]
    centroids = [np.array(centroid) for centroid in centroids]

    number_of_clusters = len(centroids)
    allotted_clusters = [0] * len(samples)

    has_changed = True
    while has_changed:
        has_changed = False

        for i, sample in enumerate(samples):
            centroid_index = min(range(number_of_clusters), key=lambda a: metric(centroids[a], sample))
            if centroid_index != allotted_clusters[i]:
                has_changed = True
            allotted_clusters[i] = centroid_index

        buckets = [[] for _ in range(number_of_clusters)]
        for feature_vector, cluster_index in zip(samples, allotted_clusters):
            buckets[cluster_index].append(feature_vector)
        for i in range(number_of_clusters):
            centroids[i] = np.mean(buckets[i], axis=0)
        del buckets

    # print(allotted_clusters)
    # print(np.around(centroids, 3))

    return centroids


class TestKMeans(unittest.TestCase):
    def test_tutorial10_question1(self):
        samples = [(-1, 3), (1, 4), (0, 5), (4, -1), (3, 0), (5, 1)]
        centroids = [(-1, 3), (5, 1)]
        output = k_means(samples, centroids, euclidean_distance)
        target = np.array([[0, 4], [4, 0]])
        self.assertTrue(np.array_equal(output, target))
