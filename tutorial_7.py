import numpy as np
from algorithms.karhunen_loeve_transform import KarhunenLoeveTransform


def question_4():
    samples = [
        np.array([[1], [2], [1]]),
        np.array([[2], [3], [1]]),
        np.array([[3], [5], [1]]),
        np.array([[2], [2], [1]])
    ]
    proj = KarhunenLoeveTransform(samples)
    for sample in samples:
        print(proj.project(sample))


if __name__ == "__main__":
    question_4()
