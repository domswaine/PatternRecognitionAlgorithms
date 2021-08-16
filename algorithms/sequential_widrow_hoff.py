import numpy as np


def sequential_widrow_hoff(Y, b, weights, learning_rate=0.1, epochs=3):
    print(Y)
    print(b)
    print(weights)

    for epoch in range(1, epochs + 1):
        print("Epoch %i - Sequential Widrow Hoff" % epoch)
        for sample, margin in zip(Y, b):
            prediction = float(np.matmul(sample.transpose(), weights))
            weights = weights + learning_rate * (margin - prediction) * sample
            print(" > Sample: %s, prediction: %.1f, weights: %s"
                  % (str(sample.transpose()), prediction, str(weights.transpose())))
        print("")
    return weights