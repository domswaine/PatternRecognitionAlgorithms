import numpy as np


class OjasLearningRule:
    def __init__(self, samples, weights, learning_rate=0.01, epochs=2):
        self.mu = np.mean(np.stack(samples), axis=0)
        self.weights = np.array(weights,  dtype='float64')
        self.load(samples, learning_rate, epochs)

    def load(self, samples, n, epochs):
        zero_mean_samples = [sample - self.mu for sample in samples]

        for epoch in range(1, epochs + 1):
            print("Epoch %i" % epoch)
            sigma_delta = np.zeros(self.weights.shape, dtype='float64')

            for i, x in enumerate(zero_mean_samples, 1):
                print("Sample %i" % i)
                x_t = np.transpose(x)
                print(" > x_t: ", x_t)
                y = float(np.matmul(self.weights, x))
                print(" > y = wx: ", y)
                sub = x_t - y * self.weights
                print(" > x_t - yw: ", sub)
                delta = n * y * sub
                print(" > ny(x_t - yw): ", delta)
                sigma_delta += delta

            self.weights += sigma_delta
            print("Sigma delta: ", sigma_delta)
            print("W: ", self.weights, "\n")

    def project(self, sample):
        return np.matmul(self.weights, sample - self.mu)
