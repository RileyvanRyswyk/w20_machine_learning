import numpy as np


def leastSquares(data, label):
    # Sum of squared error shoud be minimized
    #
    # INPUT:
    # data        : Training inputs  (num_samples x dim)
    # label       : Training targets (num_samples x 1)
    #
    # OUTPUT:
    # weights     : weights   (dim x 1)
    # bias        : bias term (scalar)

    N, D = data.shape[0], data.shape[1]
    # Extend each datapoint x as [1, x]
    # (Trick to avoid modeling the bias term explicitly)
    X = np.column_stack((np.ones((N, 1)), data))
    
    # compute (X^T * X)^-1 * X^T * T
    W = np.linalg.inv(X.T @ X) @ X.T @ label
    
    # extract outputs
    weight = W[1:]
    bias = W[0]

    return weight, bias
