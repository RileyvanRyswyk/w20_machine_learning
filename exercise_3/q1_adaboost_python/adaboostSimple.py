import numpy as np
from numpy.random import choice
from simpleClassifier import simpleClassifier

def adaboostSimple(X, Y, K, nSamples):
    # Adaboost with decision stump classifier as weak classifier
    #
    # INPUT:
    # X         : training examples (numSamples x numDim)
    # Y         : training labels (numSamples x 1)
    # K         : number of weak classifiers to select (scalar) 
    #             (the _maximal_ iteration count - possibly abort earlier
    #              when error is zero)
    # nSamples  : number of training examples which are selected in each round (scalar)
    #             The sampling needs to be weighted!
    #             Hint - look at the function 'choice' in package numpy.random
    #
    # OUTPUT:
    # alphaK 	: voting weights (K x 1) - for each round
    # para		: parameters of simple classifier (K x 3) - for each round
    #           : dimension 1 is j
    #           : dimension 2 is theta
    #           : dimension 3 is parity variable

    n, dim = X.shape[0], X.shape[1]

    # initialize weights
    W = 1/n * np.ones(n)
    para = []
    alphaK = []

    # compute k weak classifiers
    for i in range(K):
        # extract a random subset and train it
        subset = choice(n, nSamples, p=W)
        d_hat, theta, p_hat = simpleClassifier(X[subset], Y[subset])
        para.append([d_hat, theta, p_hat])

        # estimate weighted error
        e_k = 0
        for j in range(n):
            if (p_hat * X[j][d_hat] > p_hat * theta) != (Y[j] == 1):
                e_k += W[j]
        # normalize e_k
        e_k /= np.sum(W)

        # calculate a_k
        alphaK.append(1/2 * np.log((1 - e_k) / e_k))

        #check for terminate (error == 0)
        if e_k == 0:
            break

        # update weighting coefficients
        for j in range(n):
            if (p_hat * X[j][d_hat] > p_hat * theta) != (Y[j] == 1):
                W[j] = W[j] * np.exp(alphaK[i])
        #normalize W
        W = W / np.sum(W)

    return alphaK, para
