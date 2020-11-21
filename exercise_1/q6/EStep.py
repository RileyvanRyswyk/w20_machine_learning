import numpy as np
import math as math
from getLogLikelihood import getLogLikelihood


def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    if type(means) != np.ndarray:
      means = np.array(means)

    if type(X) != np.ndarray:
      X = np.array(X)

    # initialize outputs and parameters
    N, D, K = X.shape[0], X.shape[1], len(weights)
    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    gamma = np.zeros((N, K))
    
    # calculate normalization factors for k gaussians 
    # and calculate Covariance inverses for k gaussians
    # we will need to reshape the covariance matrix to KxDxD from DxDxK
    covar_trans = covariances.transpose(2, 0, 1)
    norm_factors = 1 / (np.linalg.det(covar_trans) ** (1 / 2))
    norm_factors *= 1 / (2 * math.pi) ** (D / 2)
    inverses = np.linalg.inv(covar_trans)

    # iterate over the data points to determine their responsibilities
    for n, data_point in enumerate(X):

      # iterate over each K gaussian and calculate the responsibility of component k for x_n
      for k, gaussian_weight in enumerate(weights):

        # x_n - mu_k
        y = (data_point - means[k,:])

        # exp^{-1/2 * (x_n-u_k)^T * (covariance_k)^-1 * (x_n-u_k)}
        p = norm_factors[k] * math.exp(-0.5 * y.T @ inverses[k,:,:] @ y)

        gamma[n, k] = gaussian_weight * p

    # normalize the rows of gamma, i.e. normalize the responsibilities
    row_sums = gamma.sum(axis=1) 
    gamma = gamma / row_sums[:, np.newaxis]

    return [logLikelihood, gamma]
