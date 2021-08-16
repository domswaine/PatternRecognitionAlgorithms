import numpy as np


def batch_perceptron_learning_algorithm(weights, samples, learning_rate=1, epochs=3):
    # This version of the batch perceptron learning algorithm requires normalisation and augmented notation.
    weights = np.array(weights)
    samples = [np.array(sample) for sample in samples]

    for epoch in range(1, epochs + 1):
        print("Epoch %i - batch perceptron learning algorithm" % epoch)
        sigma_y = np.zeros(weights.shape)
        for sample in samples:
            prediction = float(np.matmul(weights.transpose(), sample))
            print(" > Sample: %s, prediction: %.1f" % (str(sample.transpose()), prediction))
            if prediction <= 0:
                sigma_y += sample
        weights = weights + learning_rate * sigma_y
        print(" > Sigma y: %s, new weights: %s\n" % (str(sigma_y.transpose()), str(weights.transpose())))

    return weights


def sequential_perceptron_learning_algorithm(weights, samples, learning_rate=1, epochs=3):
    # This version of the batch perceptron learning algorithm requires normalisation and augmented notation.
    weights = np.array(weights)
    samples = [np.array(sample) for sample in samples]

    for epoch in range(1, epochs + 1):
        print("Epoch %i - sequential perceptron learning algorithm" % epoch)
        for sample in samples:
            prediction = float(np.matmul(weights.transpose(), sample))
            if prediction <= 0:
                weights = weights + learning_rate * sample
            print(" > Sample: %s, prediction: %.1f, weights: %s"
                  % (str(sample.transpose()), prediction, str(weights.transpose())))
        print("")

    return weights
