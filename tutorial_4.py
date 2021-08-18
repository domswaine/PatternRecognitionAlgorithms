import numpy as np
from algorithms.activation_functions import symmetric_tangent_sigmoid_function
from algorithms.activation_functions import logarithmic_sigmoid_function


def augment(arr):
    return np.vstack([np.ones((1, arr.shape[1])), arr])


def question_4():
    # Each column represents a sample

    patterns = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])

    wji = np.array([
        [-0.7057, 1.9061, 2.6605, -1.1359],
        [0.4900, 1.9324, -0.4269, -5.1570],
        [0.9438, -5.4160, -0.3431, -0.2931]
    ])

    wj0 = np.array([
        [4.8432],
        [0.3973],
        [2.1761]
    ])

    y = np.vectorize(symmetric_tangent_sigmoid_function)(np.matmul(wji, patterns) + wj0)

    wkj = np.array([
        [-1.1444, 0.3115, -9.9812],
        [0.0106, 11.5477, 2.6479]
    ])

    wk0 = np.array([
        [2.5230],
        [2.6463]
    ])

    z = np.vectorize(logarithmic_sigmoid_function)(np.matmul(wkj, y) + wk0)
    print(z)


if __name__ == "__main__":
    question_4()
