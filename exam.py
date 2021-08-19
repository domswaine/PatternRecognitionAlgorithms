import numpy as np
from math import exp
from algorithms.sequential_widrow_hoff import batch_widrow_hoff
from algorithms.convolution import pooling
from algorithms.batch_normalisation import batch_normalisation
from algorithms.karhunen_loeve_transform import KarhunenLoeveTransform
from algorithms.ojas_learning_rule import OjasLearningRule
from algorithms.fishers_method import fishers_method_cost
from algorithms.activation_functions import symmetric_tangent_sigmoid_function


def question_2():
    y = [
        np.array([[1.0], [0.0], [1.0]]),
        np.array([[1.0], [1.0], [-1.0]]),
        np.array([[-1], [-0.5], [-0.5]]),
        np.array([[-1], [-1.0], [-0.5]])
    ]
    a = np.array([[0.4], [-1.0], [2.0]])
    b = np.array([[1.0], [1.0], [1.0], [1.0]])
    batch_widrow_hoff(y, b, a, learning_rate=0.1, epochs=2)


def question_3():
    W = np.array([
        [0.26, -0.09, -0.68, 0.48, 0.49],
        [-0.67, 0.30, -0.53, 0.19, -0.38],
        [-0.10, -0.52, -0.12, 0.18, 0.82],
        [0.11, -0.21, -0.08, 0.24, 0.94]
    ])
    x = np.array([[0.54], [0.56], [0.37], [0.12], [0.50]])
    print(np.matmul(W, x))

    numbers = [0.141, -0.5571, 0.042, 0.411]
    numbers = [exp(n) for n in numbers]
    total = sum(numbers)
    numbers = [n / total for n in numbers]
    print(numbers)


def question_4():
    x1 = np.array([
        [0.6, -0.7, -0.7, -0.3],
        [0.1, -0.4, 0.0, -1.0],
        [0.6, 0.1, 0.0, 0.2],
        [0.8, 0.4, 0.5, 0.3]
    ])
    x2 = np.array([
        [0.1, -0.9, -0.4, -0.3],
        [-1.0, 0.3, 0.3, -0.9],
        [-0.9, 0.0, -0.7, -0.2],
        [0.4, -0.8, 0.0, -0.9]
    ])
    print(pooling(x1, (3, 3), stride=1, pooling_type=max))
    print(pooling(x2, (3, 3), stride=1, pooling_type=max))


def question_5():
    x1 = np.array([
        [-0.4, 0.4, -0.7],
        [0.1, 0.9, -0.3],
        [-0.7, 0.2, -0.5]
    ])

    x2 = np.array([
        [0.4, 0.3, 0.3],
        [-0.1, 1.0, -0.8],
        [0.5, 0.8, -0.5]
    ])

    x3 = np.array([
        [0.2, 0.9, 0.4],
        [0.0, -0.5, 0.6],
        [-0.4, -0.6, -0.9]
    ])

    batch_normalisation([x1, x2, x3], beta=0.1, gamma=0.6, epsilon=0.2)


def question_6a():
    samples = [
        np.array([[2.1], [2.5], [3.6], [5.2]]),
        np.array([[4.4], [4.3], [5.6], [3.3]]),
        np.array([[6.1], [7.0], [3.8], [2.9]]),
        np.array([[2.8], [3.5], [5.3], [4.3]]),
        np.array([[4.8], [3.5], [2.2], [5.8]])
    ]
    proj = KarhunenLoeveTransform(samples)
    for sample in samples:
        print(proj.project(sample).transpose())


def question_6b():
    samples = [
        np.array([[2.1], [2.5], [3.6], [5.2]]),
        np.array([[4.4], [4.3], [5.6], [3.3]]),
        np.array([[6.1], [7.0], [3.8], [2.9]]),
        np.array([[2.8], [3.5], [5.3], [4.3]]),
        np.array([[4.8], [3.5], [2.2], [5.8]])
    ]
    initial_weights = np.array([[0.6, 0.2, 0.2, -0.4]])
    OjasLearningRule(samples, initial_weights, learning_rate=0.01, epochs=1)


def question_7():
    samples = [
        (np.array([[2.0], [0.0]]), 1),
        (np.array([[0.0], [3.0]]), 1),
        (np.array([[3.0], [3.0]]), 1),
        (np.array([[0.0], [5.0]]), 2),
        (np.array([[3.0], [1.0]]), 2)
    ]
    fishers_method_cost(samples, np.array([[1.0, 2.0]]))
    fishers_method_cost(samples, np.array([[-1.0, -1.0]]))

    print(np.matmul(np.array([[-1.0, -1.0]]), np.array([[2.0], [0.0]])))
    print(np.matmul(np.array([[-1.0, -1.0]]), np.array([[0.0], [3.0]])))
    print(np.matmul(np.array([[-1.0, -1.0]]), np.array([[3.0], [3.0]])))


def question_8():
    # Each column represents a sample

    pattern = np.array([
        [-0.1],
        [0.4]
    ])

    wji = np.array([
        [-3.0, 3.0],
        [-4.0, 2.0],
        [-5.0, -4.0]
    ])

    wj0 = np.array([
        [-1.0],
        [2.0],
        [4.0]
    ])

    y = np.vectorize(symmetric_tangent_sigmoid_function)(np.matmul(wji, pattern) + wj0)
    print(y)

    wkj = np.array([
        [-2.0, -3.0, -2.0],
        [1.0, 4.0, -3.0]
    ])

    wk0 = np.array([
        [1.0],
        [-5.0]
    ])

    z = np.matmul(wkj, y) + wk0
    print(z)


def question_10():
    x_hat = np.array([[49.0], [36.0]])
    m1 = np.array([[44.0, 15.0]])
    m2 = np.array([[108.0, 89.0]])
    m3 = np.array([[29.0, 55.0]])

    print(np.matmul(x_hat, m1))
    print(np.matmul(x_hat, m1))
    print(np.matmul(x_hat, m1))





if __name__ == "__main__":
    # question_2()
    # question_3()
    # question_4()
    question_5()
    # question_6a()
    # question_6b()
    # question_7()
    # question_8()
    # question_10()
