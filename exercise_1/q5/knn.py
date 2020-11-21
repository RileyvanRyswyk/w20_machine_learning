import numpy as np
import math as math

def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]
    
    # d is the number of data points and
    d = samples.shape[0]

    if k > d:
        raise Exception("Insufficient data: Reduce value of k")

    # initialize outputs
    pos = np.arange(-5, 5.0, 0.1)  # Returns a 100 dimensional vector
    estDensity = np.column_stack((pos, np.zeros((100, 1))))

    # solve p(x) ~ K / N / V
    # we must determine V, which is 1-d is distance along the number line
    # this distance must include k data points

    sorted_samples = np.sort(samples)

    # loop over selected x-values to output
    for index, x in enumerate(pos):

        # bounds of nearest neighbours
        upper_index = np.searchsorted(sorted_samples, x)
        lower_index = upper_index - 1
        nn = []  # nearest neighbours

        # find the k nearest neighbours
        # k shall be less than len(samples)
        while len(nn) < k:

            # compute distances
            lower_dist = np.abs(sorted_samples[lower_index] - x) if (lower_index >= 0) else math.inf
            upper_dist = np.abs(sorted_samples[upper_index] - x) if (upper_index < len(sorted_samples)) else math.inf

            # compare and select the lowest distance option
            if lower_dist <= upper_dist:
                nn.append(sorted_samples[lower_index])
                lower_index -= 1
            elif upper_dist < lower_dist:
                nn.append(sorted_samples[upper_index])
                upper_index += 1

        # V is distance to furthest neighbour to encompass k neighbours
        estDensity[index][1] = k / d / np.abs(nn[-1] - x)

    return estDensity
