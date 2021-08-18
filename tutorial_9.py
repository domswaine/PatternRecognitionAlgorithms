import numpy as np
from algorithms.ada_boost import AdaBoost


def question_1():
    classifiers = [
        lambda x: +1 if x[0] > -0.5 else -1,
        lambda x: -1 if x[0] > -0.5 else 1,
        lambda x: +1 if x[0] > 0.5 else -1,
        lambda x: -1 if x[0] > 0.5 else -1,
        lambda x: +1 if x[1] > -0.5 else -1,
        lambda x: -1 if x[1] > -0.5 else 1,
        lambda x: +1 if x[1] > 0.5 else -1,
        lambda x: -1 if x[1] > 0.5 else 1
    ]

    samples = [
        np.array([[1], [0]]),
        np.array([[-1], [0]]),
        np.array([[0], [1]]),
        np.array([[0], [-1]])
    ]

    labels = [+1, +1, -1, -1]

    AdaBoost(classifiers, samples, labels)


if __name__ == "__main__":
    question_1()
