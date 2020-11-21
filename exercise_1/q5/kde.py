import numpy as np
import math as math

def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]


    # Compute the number of samples created
    d = samples.shape[0]

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
            estimate += math.exp(-((x - sample) ** 2) / (2 * h ** 2))

        # set kde value at x
        estDensity[index][1] = estimate * normalization_factor

    return estDensity
