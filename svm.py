import numpy as np


def svm(support_vectors, support_vector_class):
    # Source: https://github.com/ThePandaJam/PNN-exam-codes/blob/main/svm.py
    w = []
    for idx in range(len(support_vectors)):
        w.append(support_vectors[idx] * support_vector_class[idx])
    w = np.array(w)
    eq_arr = []
    for idx, sv in enumerate(support_vectors):
        tmp = ((w @ sv) * support_vector_class[idx])
        tmp = np.append(tmp, [support_vector_class[idx]])
        eq_arr.append(tmp)
    eq_arr.append(np.append(support_vector_class, [0]))
    rhs_arr = [1] * len(support_vector_class)
    rhs_arr.extend([0])
    rhs_arr = np.array(rhs_arr)
    ans = rhs_arr @ np.linalg.pinv(eq_arr)
    print("lambda and w_0 values are ", ans)
    final_weight = []
    for idx in range(w.shape[0]):
        final_weight.append(w[idx] * ans[idx])
    final_weight = np.array(final_weight)
    final_weight = np.sum(final_weight, axis=0)
    print("Weights: ", final_weight)
    print("Margin: ", 2/np.linalg.norm(final_weight))


support_vectors = np.array([
    [23.0, 82.0, 46.0, 55.0],
    [65.0, 88.0, 86.0, 115.0],
    [98.0, 8.0, 24.0, 15.0],
    [99.0, 61.0, 48.0, 80.0],
    [11.0, 46.0, 29.0, 12.0]
])
support_vector_class = np.array([1, 1, 1, -1, -1])

svm(support_vectors, support_vector_class)