import numpy as np
from algorithms.perceptron_learning_algorithms import batch_perceptron_learning_algorithm
from algorithms.perceptron_learning_algorithms import sequential_perceptron_learning_algorithm
from algorithms.sequential_widrow_hoff import sequential_widrow_hoff
from algorithms.k_nearest_neighbours import k_nearest_neighbours
from algorithms.distance_metrics import euclidean_distance


def question_1():
    g = lambda x: np.matmul(np.array([[2, 1]]), x) - 5
    print(g(np.array([[1], [1]])))
    print(g(np.array([[2], [2]])))
    print(g(np.array([[3], [3]])))


def augment(x):
    return np.vstack((np.array([[1]]), x))


def question_2():
    a = np.array([[-5], [2], [1]])
    g = lambda x: np.matmul(a.transpose(), augment(x))
    print(g(np.array([[1], [1]])))
    print(g(np.array([[2], [2]])))
    print(g(np.array([[3], [3]])))


def question_3():
    g = lambda x: x[0]**2 - x[2]**2 + 2 * x[1] * x[2] + 4 * x[0] * x[1] + 3 * x[0] - 2 * x[1] + 2
    print(g([1, 1, 1]))
    print(g([-1, 0, 3]))
    print(g([-1, 0, 0]))


def question_4():
    A = np.array([[2, 1], [1, 4]])
    b = np.array([[1], [2]])
    c = -3
    g = lambda x: np.matmul(np.matmul(x.transpose(), A), x) + np.matmul(x.transpose(), b) + c
    print(g(np.array([[0], [-1]])))
    print(g(np.array([[1], [1]])))
    A = np.array([[-2, 5], [5, -8]])
    print(g(np.array([[0], [-1]])))
    print(g(np.array([[1], [1]])))


def question_5():
    A = np.array([[-3], [1], [2], [2], [2], [4]])
    g = lambda x: np.matmul(A.transpose(), augment(x))
    print(g(np.array([[0], [-1], [0], [0], [1]])))
    print(g(np.array([[1], [1], [1], [1], [1]])))


def question_6():
    weights = np.array([[-25], [6], [3]])
    samples = [
        np.array([[1], [1], [5]]), np.array([[1], [2], [5]]),
        np.array([[-1], [-4], [-1]]), np.array([[-1], [-5], [-1]])
    ]
    batch_perceptron_learning_algorithm(weights, samples)
    sequential_perceptron_learning_algorithm(weights, samples, epochs=2)


def question_9():
    weights = np.array([[1], [0], [0]])
    samples = [
        np.array([[1], [0], [2]]),
        np.array([[1], [1], [2]]),
        np.array([[1], [2], [1]]),
        np.array([[-1], [3], [-1]]),
        np.array([[-1], [2], [1]]),
        np.array([[-1], [3], [2]])
    ]
    sequential_perceptron_learning_algorithm(weights, samples, epochs=2)


def question_12():
    y = np.array([
        [1, 0, 2],
        [1, 1, 2],
        [1, 2, 1],
        [-1, 3, -1],
        [-1, 2, 1],
        [-1, 3, 2]
    ])
    y_pseudo_inverse = np.linalg.pinv(y)
    print(np.matmul(y_pseudo_inverse, np.array([[1], [1], [1], [1], [1], [1]])))
    print(np.matmul(y_pseudo_inverse, np.array([[2], [2], [2], [1], [1], [1]])))
    print(np.matmul(y_pseudo_inverse, np.array([[1], [1], [1], [2], [2], [2]])))


def question_14():
    y = [
        np.array([[1], [0], [2]]),
        np.array([[1], [1], [2]]),
        np.array([[1], [2], [1]]),
        np.array([[-1], [3], [-1]]),
        np.array([[-1], [2], [1]]),
        np.array([[-1], [3], [2]])
    ]
    a = np.array([[1], [0], [0]])
    b = np.array([[1], [1], [1], [1], [1], [1]])
    sequential_widrow_hoff(y, b, a, epochs=2)


def question_15():
    dataset = [
        (np.array([[0.15, 0.35]]), 1),
        (np.array([[0.15, 0.28]]), 2),
        (np.array([[0.12, 0.20]]), 2),
        (np.array([[0.10, 0.32]]), 3),
        (np.array([[0.06, 0.25]]), 3)
    ]
    point = np.array([[0.1, 0.25]])
    print(k_nearest_neighbours(dataset, 1, point, euclidean_distance))
    print(k_nearest_neighbours(dataset, 3, point, euclidean_distance))


if __name__ == "__main__":
    # question_1()
    # question_2()
    # question_3()
    # question_4()
    # question_5()
    # question_6()
    # question_9()
    # question_12()
    # question_14()
    # question_15()
