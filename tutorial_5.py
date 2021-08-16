import numpy as np
from algorithms.activation_functions import rectified_linear_unit as relu
from algorithms.activation_functions import leaky_rectified_linear_unit as lrelu
from algorithms.activation_functions import tahn_activation_function as tanh
from algorithms.activation_functions import heaviside_function
from algorithms.batch_normalisation import batch_normalisation
from algorithms.convolution import convolve, pooling


def question_4():
    arr = np.array([
        [1, 0.5, 0.2],
        [-1, -0.5, -0.2],
        [0.1, -0.1, 0]
    ]).astype(float)
    print(np.vectorize(relu)(arr))
    print(np.vectorize(lambda x: lrelu(x, a=0.1))(arr))
    print(np.vectorize(tanh)(arr))
    print(np.vectorize(lambda x: heaviside_function(x, threshold=0.1, h0=0.5))(arr))


def question_5():
    x1 = np.array([
        [1, 0.5, 0.2],
        [-1, -0.5, -0.2],
        [0.1, -0.1, 0]
    ])

    x2 = np.array([
        [1, -1, 0.1],
        [0.5, -0.5, -0.1],
        [0.2, -0.2, 0]
    ])

    x3 = np.array([
        [0.5, -0.5, -0.1],
        [0, -0.4, 0],
        [0.5, 0.5, 0.2]
    ])

    x4 = np.array([
        [0.2, 1, -0.2],
        [-1, -0.6, -0.1],
        [0.1, 0, 0.1]
    ])

    batch_normalisation([x1, x2, x3, x4], beta=0, gamma=1, epsilon=0.1)


def question_6():
    x1 = np.array([
        [0.2, 1, 0],
        [-1, 0, -0.1],
        [0.1, 0, 0.1]
    ])

    x2 = np.array([
        [1, 0.5, 0.2],
        [-1, -0.5, -0.2],
        [0.1, -0.1, 0]
    ])

    h1 = np.array([
        [1, -0.1],
        [1, -0.1]
    ])

    h2 = np.array([
        [0.5, 0.5],
        [-0.5, -0.5]
    ])

    print(convolve(x1, h1) + convolve(x2, h2))
    print(convolve(x1, h1, padding=1) + convolve(x2, h2, padding=1))
    print(convolve(x1, h1, padding=1, stride=2) + convolve(x2, h2, padding=1, stride=2))
    print(convolve(x1, h1, padding=0, stride=1, dilation=2) + convolve(x2, h2, padding=0, stride=1, dilation=2))


def question_7():
    x1 = np.array([
        [0.2, 1, 0],
        [-1, 0, -0.1],
        [0.1, 0, 0.1]
    ])

    x2 = np.array([
        [1, 0.5, 0.2],
        [-1, -0.5, -0.2],
        [0.1, -0.1, 0]
    ])

    x3 = np.array([
        [0.5, -0.5, -0.1],
        [0, -0.4, 0],
        [0.5, 0.5, 0.2]
    ])

    print(x1 + -1 * x2 + 0.5 * x3)


def question_8():
    x1 = np.array([
        [0.2, 1, 0, 0.4],
        [-1, 0, -0.1, -0.1],
        [0.1, 0, -1, -0.5],
        [0.4, -0.7, -0.5, 1]
    ])
    avg = lambda lst: sum(lst) / len(lst)
    print(pooling(x1, (2, 2), stride=2, pooling_type=avg))
    print(pooling(x1, (2, 2), stride=2, pooling_type=max))
    print(pooling(x1, (3, 3), stride=1, pooling_type=max))


if __name__ == "__main__":
    # question_4()
    # question_5()
    # question_6()
    # question_7()
    question_8()
