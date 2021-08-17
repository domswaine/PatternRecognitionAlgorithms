import numpy as np


def fishers_method_cost(labelled_samples, weights):
    cluster_points = {}
    for sample, label in labelled_samples:
        if label not in cluster_points:
            cluster_points[label] = []
        cluster_points[label].append(sample)
    cluster_points = cluster_points.values()

    centroids = []
    for cluster in cluster_points:
        centroids.append(np.mean(cluster, axis=0))

    sb = 0
    for i, centroid_a in enumerate(centroids):
        for centroid_b in centroids[i:]:
            sb += np.matmul(weights, centroid_a - centroid_b)**2

    sw = 0
    for i, cluster in enumerate(cluster_points):
        centroid = centroids[i]
        for point in cluster:
            sw += np.matmul(weights, centroid - point)**2

    cost = sb/sw

    print("Between class scatter (sb): %f" % sb)
    print("Within class scatter (sw): %f" % sw)
    print("Cost: %f" % cost)

    return cost


