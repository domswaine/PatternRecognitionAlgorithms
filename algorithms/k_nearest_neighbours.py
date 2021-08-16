import numpy as np
from collections import Counter


def k_nearest_neighbours(points, k, point, metric):
    sorted_points = sorted(points, key=lambda x: metric(point, x[0]))
    labels = Counter([label for _, label in sorted_points[:k]])
    return labels.most_common()
