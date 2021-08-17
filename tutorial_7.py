import numpy as np
from algorithms.karhunen_loeve_transform import KarhunenLoeveTransform
from algorithms.ojas_learning_rule import OjasLearningRule
from algorithms.fishers_method import fishers_method_cost


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


if __name__ == "__main__":
    # question_4()
    # question_6()
    # question_7()
    question_8()
