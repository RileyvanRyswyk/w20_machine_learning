import numpy as np
import math
from numpy.random import choice
from leastSquares import leastSquares
from eval_adaBoost_leastSquare import eval_adaBoost_leastSquare


def adaboostUSPS(X, Y, K, nSamples, percent):
    # Adaboost with least squares linear classifier as weak classifier on USPS data
    # for a high dimensional dataset
    #
    # INPUT:
    # X         : the dataset (numSamples x numDim)
    # Y         : labeling    (numSamples x 1)
    # K         : number of weak classifiers (scalar)
    # nSamples  : number of data points obtained by weighted sampling (scalar)
    #
    # OUTPUT:
    # alphaK    : voting weights (1 x k) 
    # para      : parameters of simple classifier (K x (D+1))            
    #             For a D-dim dataset each simple classifier has D+1 parameters
    # error     : training error (1 x k)

    n, dim = X.shape[0], X.shape[1]

    # Sample random a percentage of data as test data set
    testIndices = choice(n, math.ceil(percent * n), replace=False)
    testX = X[testIndices]
    testY = Y[testIndices]
    testN = testX.shape[0]
    trainIndices = np.delete(np.arange(n), testIndices, None)
    trainX = X[trainIndices]
    trainY = Y[trainIndices]
    trainN = trainX.shape[0]

    # initialize weights
    W = 1 / trainN * np.ones(trainN)
    para = []
    alphaK = []
    error = []

    # compute k weak classifiers
    for i in range(K):
        # extract a random subset and train it
        subset = choice(trainN, nSamples, p=W)
        weight, bias = leastSquares(trainX[subset], trainY[subset])
        para.append(np.concatenate((bias, weight[:, 0])))

        # estimate weighted error
        e_k = 0
        for j in range(trainN):
            if (np.dot(weight[:, 0], trainX[j]) + bias[0] > 0) != (trainY[j, 0] == 1):
                e_k += W[j]
        # normalize e_k
        e_k /= np.sum(W)

        # calculate a_k
        alphaK.append(1 / 2 * np.log((1 - e_k) / e_k))

        # check for terminate (error == 0)
        if e_k == 0:
            break

        # update weighting coefficients
        for j in range(trainN):
            if (np.dot(weight[:, 0], trainX[j]) + bias[0] > 0) != (trainY[j, 0] == 1):
                W[j] = W[j] * np.exp(alphaK[i])
        # normalize W
        W = W / np.sum(W)

        # calculate error at current K for test set
        k_error = 0
        for l in range(testN):
            sum = 0
            for k in range(i + 1):
                w0 = para[k][0]
                w1_n = para[k][1:]
                if np.dot(w1_n, testX[l]) + w0 > 0:
                    sum += alphaK[k]
                else:
                    sum -= alphaK[k]
            if np.sign(sum) != testY[l, 0]:
                k_error += 1
        error.append(k_error / testN)
    print(error)
    return [alphaK, para, error]
