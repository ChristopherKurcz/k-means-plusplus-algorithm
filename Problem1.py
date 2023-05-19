import numpy as np
import scipy as sp
import random
import math
import matplotlib.pyplot as plt

def k_init(X, k):
    """ k-means++: initialization algorithm

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    Returns
    -------
    init_centers: array (k, d)
        The initialize centers for kmeans++
    """

    # initialize first center to be a random data point
    centers = []
    centers.append(X[np.random.randint(X.shape[0])])

    for c_i in range(k-1):
        # compute squared euclidean distances of every point from nearest center
        distances = []
        sumDistances = 0
        for i in range(X.shape[0]):
            datapoint = X[i]
            dist = math.dist(datapoint, centers[0])**2
            for j in range(len(centers)):
                temp_dist = math.dist(datapoint, centers[j])**2
                dist = min(dist, temp_dist)
            distances.append(dist)
            sumDistances += dist
        
        # compute chances of every point to be the next center
        chances = []
        for i in range(X.shape[0]):
            pointDist = distances[i]
            chance = pointDist / sumDistances
            chances.append(chance)

        # choose next center based on chances for each data point        
        nextCenter = np.array(random.choices(X, weights = chances, k=1))[0]
        centers.append(nextCenter)

    # return the centers array
    return np.array(centers)


def k_means_pp(X, k, max_iter):
    """ k-means++ clustering algorithm

    step 1: call k_init() to initialize the centers
    step 2: iteratively refine the assignments

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    max_iter: int
        Maximum number of iteration

    Returns
    -------
    final_centers: array, shape (k, d)
        The final cluster centers
    """

    # use k-means++ do intialize centers
    centers = k_init(X, k)
    objective = compute_objective(X, centers)

    for i in range(max_iter):
        # run through the k-means algorithm
        dataMap = assign_data2clusters(X, centers)
        newCenters = update_centers(X, dataMap)
        newObjective = compute_objective(X, newCenters)
        if newObjective < objective:
            objective = newObjective
            centers = newCenters
        else:
           break

    return centers


def assign_data2clusters(X, C):
    """ Assignments of data to the clusters
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    Returns
    -------
    data_map: array, shape(n, k)
        The binary matrix A which shows the assignments of data points (X) to
        the input centers (C).
    """
    dataMap = []
    k = C.shape[0]

    for i in range(X.shape[0]):
        # for each data point, find the center with the minimium distance to it
        datapoint = X[i]
        closestCenterIndex = 0
        closestCenterDist = math.dist(datapoint, C[0])**2
        for j in range(k):
            tempDist = math.dist(datapoint, C[j])**2
            if tempDist < closestCenterDist:
                closestCenterIndex = j
                closestCenterDist = tempDist
        
        # create an array with a 1 on the index of the closest center
        closestCenter = np.zeros(k, dtype=int)
        closestCenter[closestCenterIndex] = 1
        dataMap.append(closestCenter)

    dataMap = np.array(dataMap)
    return dataMap


def compute_objective(X, C):
    """ Compute the clustering objective for X and C
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    Returns
    -------
    accuracy: float
        The objective for the given assigments
    """

    result = 0

    for i in range(X.shape[0]):
        # compute the distance to the closest centroid for each data point
        minDistance = math.dist(X[i], C[0])**2
        for j in range(C.shape[0]):
            tempDist = math.dist(X[i], C[j])**2
            if tempDist < minDistance:
                minDistance = tempDist
        
        # sum up all the distances
        result += minDistance

    return result


def update_centers(X, D):
    """ Compute the new centers of each cluster
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    D: array, shape(n, k)
        The binary matrix A which shows the assignments of data points (X) to
        the input centers (C).

    Returns
    -------
    new_centers: array, shape (k, d)
        The final cluster centers
    """
    n = X.shape[0]
    k = len(D[0])
    d = len(X[0])

    clusterSums = [[0.0 for col in range(d)] for row in range(k)]
    clusterNums = [0.0 for i in range(k)]

    for dataPoint in range(n):
        # get the index of the cluster that the data point is inside of
        clusterIndex = np.where(D[dataPoint]==1)[0][0]

        # keep track of the number of data points in each cluster
        clusterNums[clusterIndex] += 1

        # add the value of the features of the data point to the sum for its cluster
        for feat in range(d):
            clusterSums[clusterIndex][feat] += X[dataPoint][feat]
    
    # find the true centers of each cluster
    trueCenters = np.zeros((k,d))
    for c in range(k):
        for f in range(d):
            trueCenters[c][f] = clusterSums[c][f] / clusterNums[c]
    
    newCenters = np.array(trueCenters)

    return newCenters
