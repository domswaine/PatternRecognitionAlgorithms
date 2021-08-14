# python -m unittest leader_follower_algorithm

import unittest
import numpy as np
from distance_metrics import euclidean_distance


class LeaderFollowerAlgorithm:
    def __init__(self, samples, theta, n, selection_order):
        self.samples = [np.array(sample) for sample in samples]
        self.centroids = []
        self.theta = theta
        self.n = n
        self.selection_order = selection_order

    def index_of_nearest_cluster(self, sample):
        number_of_clusters = len(self.centroids)
        return min(range(number_of_clusters), key=lambda i: euclidean_distance(self.centroids[i], sample))

    def learn(self):
        for index in self.selection_order:
            sample = self.samples[index]

            if len(self.centroids) == 0:
                self.centroids.append(sample)

            else:
                j = self.index_of_nearest_cluster(sample)
                if euclidean_distance(sample, self.centroids[j]) < self.theta:
                    self.centroids[j] = self.centroids[j] + self.n * (sample - self.centroids[j])
                else:
                    self.centroids.append(sample)

    def get_centroids(self):
        return self.centroids

    def classify_sample(self, sample):
        centroid_index = self.index_of_nearest_cluster(sample)
        return self.centroids[centroid_index]


class TestLeaderFollowerAlgorithm(unittest.TestCase):
    def test_tutorial10_question4(self):
        samples = [(-1, 3), (1, 4), (0, 5), (4, -1), (3, 0), (5, 1)]
        model = LeaderFollowerAlgorithm(samples, 3, 0.5, [2, 0, 0, 4, 5])
        model.learn()
        m1, m2 = model.get_centroids()
        self.assertTrue(np.allclose(m1, np.array([-0.75, 3.5])))
        self.assertTrue(np.allclose(m2, np.array([4, 0.5])))
        self.assertTrue(np.allclose(model.classify_sample(np.array([0, -2])), m2))
