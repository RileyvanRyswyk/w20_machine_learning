import numpy as np
import math
from numpy.random import choice
from simpleClassifier import simpleClassifier
from eval_adaBoost_simpleClassifier import eval_adaBoost_simpleClassifier

def adaboostCross(X, Y, K, nSamples, percent):
    # Adaboost with an additional cross validation routine
    #
    # INPUT:
    # X         : training examples (numSamples x numDims )
    # Y         : training lables (numSamples x 1)
    # K         : number of weak classifiers to select (scalar)
    #             (the _maximal_ iteration count - possibly abort earlier)
    # nSamples  : number of training examples which are selected in each round. (scalar)
    #             The sampling needs to be weighted!
    # percent   : percentage of the data set that is used as test data set (scalar)
    #
    # OUTPUT:
    # alphaK    : voting weights (K x 1)
    # para      : parameters of simple classifier (K x 2)
    # testX     : test dataset (numTestSamples x numDim)
    # testY     : test labels  (numTestSamples x 1)
    # error	    : error rate on validation set after each of the K iterations (K x 1)

    n, dim = X.shape[0], X.shape[1]

    # Randomly sample a percentage of the data as test data set
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
        d_hat, theta, p_hat = simpleClassifier(trainX[subset], trainY[subset])
        para.append([d_hat, theta, p_hat])

        # estimate weighted error
        e_k = 0
        for j in range(trainN):
            if (p_hat * trainX[j][d_hat] > p_hat * theta) != (trainY[j] == 1):
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
            if (p_hat * trainX[j][d_hat] > p_hat * theta) != (trainY[j] == 1):
                W[j] = W[j] * np.exp(alphaK[i])
        # normalize W
        W = W / np.sum(W)

        # calculate error at current K for test set
        k_error = 0
        for l in range(testN):
            sum = 0
            for k in range(i + 1):
                d_hat = para[k][0]
                theta = para[k][1]
                p_hat = para[k][2]
                if p_hat * testX[l][d_hat] > p_hat * theta:
                    sum += alphaK[k]
                else:
                    sum -= alphaK[k]
            if np.sign(sum) != testY[l]:
                k_error += 1
        error.append(k_error / testN)

    return alphaK, para, testX, testY, error

