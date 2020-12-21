import numpy as np


def simpleClassifier(X, Y):
    # Select a simple classifier
    #
    # INPUT:
    # X         : training examples (numSamples x numDim)
    # Y         : training lables (numSamples x 1)
    #
    # OUTPUT:
    # theta 	: threshold value for the decision (scalar)
    # j 		: the dimension to "look at" (scalar)
    # p_hat     : the parity variable for the inequality

    n, dim = X.shape[0], X.shape[1]
    j, theta, minCount = None, None, np.inf

    # Evaluate classifier for each data point value over each axis
    # Run time is O(N^2*D)
    for d in range(dim):
        for p in (-1, 1):
            for threshold in range(n):
                count = 0
                for i in range(n):
                    if (p * X[i][d] > p * X[threshold][d]) != (Y[i] == 1):
                        count += 1
                if count < minCount:
                    j, theta, p_hat, minCount = d, X[threshold][d], p, count

    return j, theta, p_hat

