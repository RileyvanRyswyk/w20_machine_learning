import numpy as np
import numbers as numbers
import math as math
import matplotlib.pyplot as plt


def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    # samples : DxN matrix of data points
    # h : (half) window size/radius of kernel
    # Output
    # estDensity : estimated density in the range of [-5,5]

    # validate inputs
    if type(samples) != np.ndarray or samples.ndim != 2:
        raise Exception("Samples must be a matrix")
    if isinstance(h, numbers.Number) is False or h <= 0:
        raise Exception("h must be a positive number")

    # assuming d is the number of data points and
    # assuming dim = 1 (univariate KDE)
    d, dim = samples.shape[0], samples.shape[1]

    if dim > 1:
        raise Exception("Only univariate data supported")

    # initialize outputs
    pos = np.arange(-5, 5.0, 0.1)  # Returns a 100 dimensional vector
    estDensity = np.column_stack((pos, np.zeros((100, 1))))

    # constant term for normalization in gaussian kernel
    normalization_factor = 1 / d * 1 / math.sqrt(2 * math.pi * (h ** 2))

    # loop over selected x-values to output
    for index, x in enumerate(pos):
        estimate = 0

        # loop over input data points to calculate the kde at x
        for sample in samples:
            estimate += math.exp(-((x - sample[0]) ** 2) / (2 * h ** 2))

        # set kde value at x
        estDensity[index][1] = estimate * normalization_factor

    return estDensity


def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    # samples : DxN matrix of data points
    # k : number of neighbors
    # Output
    # estDensity : estimated density in the range of [-5, 5]

    # validate inputs
    if type(samples) != np.ndarray or samples.ndim != 2:
        raise Exception("Samples must be a matrix")
    if isinstance(k, int) is False or k <= 0:
        raise Exception("k must be a positive number")

    # assuming d is the number of data points and
    # assuming dim = 1 (univariate KDE)
    d, dim = samples.shape[0], samples.shape[1]

    if dim > 1:
        raise Exception("Only univariate data supported")
    if k > d:
        raise Exception("Insufficient data: Reduce value of k")

    # initialize outputs
    pos = np.arange(-5, 5.0, 0.1)  # Returns a 100 dimensional vector
    estDensity = np.column_stack((pos, np.zeros((100, 1))))

    # solve p(x) ~ K / N / V
    # we must determine V, which is 1-d is distance along the number line
    # this distance must include k data points

    sorted_samples = np.sort(samples[:, 0])

    # loop over selected x-values to output
    for index, x in enumerate(pos):
        estimate = 0

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


def test_kde_knn():
    # test kde method with discrete data randomly sampled from a mixture of gaussians
    samples = np.concatenate((
        np.random.normal(-3, 1.5, 50),
        np.random.normal(2, 2.5, 50)
    )).reshape(100, 1)

    results = kde(samples, 0.75)
    results1 = knn(samples, 30)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    color1 = 'tab:green'
    ax1.set_xlabel('parameter')
    ax1.set_ylabel('expectation', color=color)
    ax1.plot(results[0:100, 0], results[0:100, 1], color=color)
    ax1.plot(results1[0:100, 0], results1[0:100, 1], color=color1)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('data point count', color=color)
    ax2.hist(samples[0:100, 0], bins=20, color=color, alpha=0.5)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.show()

# run test
test_kde_knn()
