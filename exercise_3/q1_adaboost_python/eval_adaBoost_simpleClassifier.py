import numpy as np


def eval_adaBoost_simpleClassifier(X, alphaK, para):
    # INPUT:
    # para	: parameters of simple classifier (K x 3) - for each round
    #           : dimension 1 is j
    #           : dimension 2 is theta
    #           : dimension 3 is p
    # alphaK    : classifier voting weights (K x 1)
    # X         : test data points (numSamples x numDim)
    #
    # OUTPUT:
    # classLabels: labels for data points (numSamples x 1)
    # result     : weighted sum of all the K classifier (numSamples x 1)

    n, dim, K = X.shape[0], X.shape[1], len(alphaK)
    classLabels = []
    result = []

    # compute H(X) for each data point
    for i in range(n):
        sum = 0
        for k in range(K):
            d_hat = para[k][0]
            theta = para[k][1]
            p_hat = para[k][2]
            if p_hat * X[i][d_hat] > p_hat * theta:
                sum += alphaK[k]
            else:
                sum -= alphaK[k]
        result.append(sum)
        classLabels.append(int(np.sign(sum)))

    return np.array(classLabels), result
