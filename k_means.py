# python -m unittest k_means

import unittest
import numpy as np
from distance_metrics import euclidean_distance


class KMeans:
    def __init__(self, samples, centroids, metric):
        self.samples = [np.array(sample) for sample in samples]
        self.centroids = [np.array(centroid) for centroid in centroids]
        self.metric = metric

    def learn(self):
        number_of_clusters = len(self.centroids)
        allotted_clusters = [0] * len(self.samples)

        has_changed = True
        while has_changed:
            has_changed = False

            for i, sample in enumerate(self.samples):
                centroid_index = min(range(number_of_clusters), key=lambda a: self.metric(self.centroids[a], sample))
                if centroid_index != allotted_clusters[i]:
                    has_changed = True
                allotted_clusters[i] = centroid_index

            buckets = [[] for _ in range(number_of_clusters)]
            for feature_vector, cluster_index in zip(self.samples, allotted_clusters):
                buckets[cluster_index].append(feature_vector)
            for i in range(number_of_clusters):
                self.centroids[i] = np.mean(buckets[i], axis=0)
            del buckets

    def get_centroids(self):
        return self.centroids


class TestKMeans(unittest.TestCase):
    def test_tutorial10_question1(self):
        samples = [(-1, 3), (1, 4), (0, 5), (4, -1), (3, 0), (5, 1)]
        centroids = [(-1, 3), (5, 1)]
        model = KMeans(samples, centroids, euclidean_distance)
        model.learn()

        target = np.array([[0, 4], [4, 0]])
        self.assertTrue(np.array_equal(model.get_centroids(), target))
