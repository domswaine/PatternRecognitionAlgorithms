# python -m unittest competitive_learning_algorithm

import unittest
import numpy as np
from distance_metrics import euclidean_distance


class CompetitiveLearningAlgorithm:
    def __init__(self, samples, centroids, n, selection_order):
        self.samples = [np.array(sample) for sample in samples]
        self.centroids = [np.array(centroid) for centroid in centroids]
        self.selection_order = selection_order
        self.n = n
        self.number_of_clusters = len(centroids)

    def index_of_nearest_cluster(self, sample):
        return min(range(self.number_of_clusters), key=lambda i: euclidean_distance(self.centroids[i], sample))

    def learn(self):
        for index in self.selection_order:
            sample = self.samples[index]
            j = self.index_of_nearest_cluster(sample)
            self.centroids[j] = self.centroids[j] + self.n * (sample - self.centroids[j])

    def get_centroids(self):
        return self.centroids

    def classify_sample(self, sample):
        centroid_index = self.index_of_nearest_cluster(sample)
        return self.centroids[centroid_index]


class TestCompetitiveLearningAlgorithm(unittest.TestCase):
    def test_tutorial10_question3(self):
        samples = [(-1, 3), (1, 4), (0, 5), (4, -1), (3, 0), (5, 1)]
        centroids = [(-0.5, 1.5), (0, 2.5), (1.5, 0)]
        cla = CompetitiveLearningAlgorithm(samples, centroids, 0.1, [2, 0, 0, 4, 5])
        cla.learn()
        m1, m2, m3 = cla.get_centroids()
        self.assertTrue(np.allclose(m1, np.array([-0.5, 1.5])))
        self.assertTrue(np.allclose(m2, np.array([-0.19, 2.7975])))
        self.assertTrue(np.allclose(m3, np.array([1.985, 0.1])))
        self.assertTrue(np.allclose(cla.classify_sample(np.array([0, -2])), m3))
