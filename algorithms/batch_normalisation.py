import numpy as np


def batch_normalisation(samples, beta, gamma, epsilon):
    tensor = np.stack(tuple(samples))
    var = np.var(tensor, axis=0)
    avg = np.mean(tensor, axis=0)
    for i, sample in enumerate(samples, 1):
        norm = beta + gamma * ((sample - avg) / (np.sqrt(var + epsilon)))
        print("Sample %i\n" % i, norm, "\n")
