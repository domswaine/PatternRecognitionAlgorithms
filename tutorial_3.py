import numpy as np
from algorithms.activation_functions import heaviside_function
from algorithms.delta_learning_algorithms import sequential_delta_learning_rule
from algorithms.delta_learning_algorithms import batch_delta_learning_rule


def question_2():
    w = np.array([[0.1, -0.5, 0.4]])
    x1 = np.array([[0.1], [-0.5], [0.4]])
    print(heaviside_function(np.matmul(w, x1)))
    x2 = np.array([[0.1], [0.5], [0.4]])
    print(heaviside_function(np.matmul(w, x2)))


def question_3():
    samples = [np.array([[1], [0]]), np.array([[1], [1]])]
    labels = [1, 0]
    weights = np.array([[-1.5, 2]])
    sequential_delta_learning_rule(weights, samples, labels)


def question_4():
    samples = [np.array([[1], [0]]), np.array([[1], [1]])]
    labels = [1, 0]
    weights = np.array([[-1.5, 2]])
    batch_delta_learning_rule(weights, samples, labels, epochs=7)


def question_5():
    samples = [
        np.array([[1], [0], [0]]),
        np.array([[1], [0], [1]]),
        np.array([[1], [1], [0]]),
        np.array([[1], [1], [1]])
    ]
    labels = [0, 0, 0, 1]
    weights = np.array([[0.5, 1, 1]])
    sequential_delta_learning_rule(weights, samples, labels, epochs=5)


def question_6():
    samples = [
        np.array([[1], [0], [2]]),
        np.array([[1], [1], [2]]),
        np.array([[1], [2], [1]]),
        np.array([[1], [-3], [1]]),
        np.array([[1], [-2], [-1]]),
        np.array([[1], [-3], [-2]])
    ]
    labels = [1, 1, 1, 0, 0, 0]
    weights = np.array([[1, 0, 0]])
    sequential_delta_learning_rule(weights, samples, labels, epochs=3)


if __name__ == "__main__":
    # question_2()
    # question_3()
    # question_4()
    # question_5()
    question_6()