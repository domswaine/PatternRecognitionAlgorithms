import numpy as np
from algorithms.activation_functions import heaviside_function


def sequential_delta_learning_rule(weights, samples, labels, learning_rate=1, epochs=3):
    # This version of the batch perceptron learning algorithm requires augmented notation.

    for epoch in range(1, epochs + 1):
        print("Epoch %i - sequential delta learning rule" % epoch)
        for sample, label in zip(samples, labels):
            prediction = heaviside_function(np.matmul(weights, sample))
            weights = weights + learning_rate * (label - prediction) * sample.transpose()
            print(" > Sample: %s, prediction: %.1f, weights: %s"
                  % (str(sample.transpose()), prediction, str(weights)))
        print("")

    return weights


def batch_delta_learning_rule(weights, samples, labels, learning_rate=1, epochs=3):
    # This version of the batch perceptron learning algorithm requires augmented notation.

    for epoch in range(1, epochs + 1):
        print("Epoch %i - sequential delta learning rule" % epoch)
        delta = np.zeros(weights.shape)

        for sample, label in zip(samples, labels):
            prediction = heaviside_function(np.matmul(weights, sample))
            delta += (label - prediction) * sample.transpose()
            print(" > Sample: %s, prediction: %.1f" % (str(sample.transpose()), prediction))

        weights = weights + delta
        print(" > New weights: %s" % weights)
        print("")

    return weights
