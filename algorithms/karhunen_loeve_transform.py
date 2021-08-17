import numpy as np


class KarhunenLoeveTransform:
    def __init__(self, samples):
        self.mu = None
        self.v_hat_t = None
        self.load(samples)

    def load(self, samples):
        number_of_samples = len(samples)
        dimensionality = samples[0].size

        self.mu = np.mean(np.stack(samples), axis=0)
        zero_mean_data = [sample - self.mu for sample in samples]

        covariance_matrix = np.zeros((dimensionality, dimensionality))
        for zero_mean_sample in zero_mean_data:
            covariance_matrix += np.matmul(zero_mean_sample, zero_mean_sample.transpose())
        covariance_matrix = covariance_matrix / number_of_samples

        E, V = np.linalg.eig(covariance_matrix)
        indexes = (-E).argsort()[:dimensionality - 1]
        self.v_hat_t = V[:, indexes].transpose()

    def project(self, sample):
        return np.matmul(self.v_hat_t, sample - self.mu)