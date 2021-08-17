import numpy as np
from algorithms.karhunen_loeve_transform import KarhunenLoeveTransform
from algorithms.ojas_learning_rule import OjasLearningRule
from algorithms.fishers_method import fishers_method_cost
from algorithms.activation_functions import heaviside_function


def question_4():
    samples = [
        np.array([[1], [2], [1]]),
        np.array([[2], [3], [1]]),
        np.array([[3], [5], [1]]),
        np.array([[2], [2], [1]])
    ]
    proj = KarhunenLoeveTransform(samples)
    for sample in samples:
        print(proj.project(sample).transpose())


def question_6():
    samples = [
        np.array([[0], [1]]),
        np.array([[3], [5]]),
        np.array([[5], [4]]),
        np.array([[5], [6]]),
        np.array([[8], [7]]),
        np.array([[9], [7]])
    ]
    proj = KarhunenLoeveTransform(samples)
    for sample in samples:
        print(proj.project(sample).transpose())


def question_7():
    samples = [
        np.array([[0], [1]]),
        np.array([[3], [5]]),
        np.array([[5], [4]]),
        np.array([[5], [6]]),
        np.array([[8], [7]]),
        np.array([[9], [7]])
    ]
    initial_weights = np.array([[-1.0, 0.0]])
    OjasLearningRule(samples, initial_weights)


def question_8():
    samples = [
        (np.array([[1], [2]]), 1),
        (np.array([[2], [1]]), 1),
        (np.array([[3], [3]]), 1),
        (np.array([[6], [5]]), 2),
        (np.array([[7], [8]]), 2)
    ]
    fishers_method_cost(samples, np.array([[-1, 5]]))
    fishers_method_cost(samples, np.array([[2, -3]]))


def question_9():
    v = np.array([
        [-0.62, 0.44, -0.91],
        [-0.81, -0.09, 0.02],
        [0.74, -0.91, -0.60],
        [-0.82, -0.92, 0.71],
        [-0.26, 0.68, 0.15],
        [0.80, -0.94, -0.83]
    ])

    x = np.array([
        [1, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 1, 0, 1]
    ])

    y = np.vectorize(heaviside_function)(np.matmul(v, x))
    print(y)

    print(np.matmul(np.array([[0, 0, 0, -1, 0, 0, 2]]), y))


if __name__ == "__main__":
    # question_4()
    # question_6()
    # question_7()
    # question_8()
    question_9()
