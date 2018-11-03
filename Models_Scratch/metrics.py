import numpy as np

def euclidean_loop(a, b):
    dist = 0
    for i in range(len(a)):
        dist += np.square(a[i]-b[i])
    return np.sqrt(dist)

def manhattan(a, b):
    dist = 0
    for i in range(len(a)):
        dist += np.abs(a[i]-b[i])
    return dist

def euclidean(a, b):
    return np.sqrt(((a-b)**2).sum(axis=0))

def manhattan(a, b):
    return np.abs((a-b).sum(axis=0))

def minkowski(a, b, p):
    return ((np.abs(a-b)**p).sum(axis=0))**(1/p)
