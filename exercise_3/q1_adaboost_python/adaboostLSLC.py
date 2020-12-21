import numpy as np
from numpy.random import choice
from leastSquares import leastSquares

def adaboostLSLC(X, Y, K, nSamples):
    # Adaboost with least squares linear classifier as weak classifier
    # for a D-dim dataset
    #
    # INPUT:
    # X         : the dataset (numSamples x numDim)
    # Y         : labeling    (numSamples x 1)
    # K         : number of weak classifiers (iteration number of Adaboost) (scalar)
    # nSamples  : number of data which are weighted sampled (scalar)
    #
    # OUTPUT:
    # alphaK    : voting weights (K x 1)
    # para      : parameters of least square classifier (K x 3) 
    #             For a D-dim dataset each least square classifier has D+1 parameters
    #             w0, w1, w2........wD

    n, dim = X.shape[0], X.shape[1]

    # initialize weights
    W = 1 / n * np.ones(n)
    para = []
    alphaK = []

    # compute k weak classifiers
    for i in range(K):
        # extract a random subset and train it
        subset = choice(n, nSamples, p=W)
        weight, bias = leastSquares(X[subset], Y[subset])
        para.append([bias, weight[0], weight[1]])

        # estimate weighted error
        e_k = 0
        for j in range(n):
            if (np.dot(weight[:, 0], X[j]) + bias[0] > 0) != (Y[j][0] == 1):
                e_k += W[j]
        # normalize e_k
        e_k /= np.sum(W)

        # calculate a_k
        alphaK.append(1 / 2 * np.log((1 - e_k) / e_k))

        # check for terminate (error == 0)
        if e_k == 0:
            break

        # update weighting coefficients
        for j in range(n):
            if (np.dot(weight[:, 0], X[j]) + bias[0] > 0) != (Y[j][0] == 1):
                W[j] = W[j] * np.exp(alphaK[i])
        # normalize W
        W = W / np.sum(W)

    return [alphaK, para]
