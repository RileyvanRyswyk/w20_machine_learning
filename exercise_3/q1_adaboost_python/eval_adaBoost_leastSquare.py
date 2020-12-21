import numpy as np

def eval_adaBoost_leastSquare(X, alphaK, para):
    # INPUT:
    # para		: parameters of simple classifier (K x (D +1)) 
    #           : dimension 1 is w0
    #           : dimension 2 is w1
    #           : dimension 3 is w2
    #             and so on
    # alphaK    : classifier voting weights (K x 1)
    # X         : test data points (numSamples x numDim)
    #
    # OUTPUT:
    # classLabels: labels for data points (numSamples x 1)
    # result     : weighted sum of all the K classifier (scalar)

    n, dim, K = X.shape[0], X.shape[1], len(alphaK)
    classLabels = []
    result = []

    # compute H(X) for each data point
    for i in range(n):
        sum = 0
        for k in range(K):
            w0 = para[k][0]
            w1 = para[k][1]
            w2 = para[k][2]
            if w0 + w1 * X[i][0] + w2 * X[i][1] > 0:
                sum += alphaK[k]
            else:
                sum -= alphaK[k]
        result.append(sum)
        classLabels.append(int(np.sign(sum)))

    return [np.array(classLabels), result]

