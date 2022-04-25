import inline as inline
import pandas as pd
import numpy as np

# Reading file into program.


def read_csv():

    _file = pd.read_csv("Data/iris_cluster_data.txt", header=None)

    return _file


# Initializing random centroids.


def initialize_centroids(data, k):

    n_dims = data.shape[1]
    centroids_min = data.min().min()
    centroids_max = data.max().max()
    _centroids = []

    for c in range(k):
        centroid = np.random.uniform(centroids_min, centroids_max, n_dims)
        _centroids.append(centroid)
        # print(centroid)

    _centroids = pd.DataFrame(_centroids, columns=data.columns)
    print(_centroids)
    return _centroids


file = read_csv()
centroids = initialize_centroids(file, 3)




