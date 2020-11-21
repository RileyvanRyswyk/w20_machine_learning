import numpy as np
from getLogLikelihood import getLogLikelihood


def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    # initialize outputs & params
    D, K, N = X.shape[1], gamma.shape[1], gamma.shape[0]
    means = np.zeros((K, D))
    covariances = np.zeros((D, D, K))

    # soft number of samples associated with each gaussian
    # i.e. columnwise sum of gamma, yielding 1xK vector
    soft_n = np.sum(gamma, axis = 0)

    # soft number of samples / total number of samples  
    weights = soft_n / N

    # calculate new means & covariances
    for k in range(0, K):
      # calculate the new mean of this k-th gaussian by considering the
      # responsilbity weighted average of the data points 
      means[k,:] = 1 / soft_n[k] * np.sum((np.diag(gamma[:,k]) @ X), axis = 0)

      # initialize covariance matrix
      cov_k = np.zeros((D, D))

      # iterate over N data points to add their contributions
      for n in range(0, N):
        
        # calculate covariance contribution
        x_hat = X[n, :] - means[k,:]
        cov_k += gamma[n, k] * np.outer(x_hat, x_hat)
      
      # set new covariance for k-th gaussian
      covariances[:, :, k] = cov_k / soft_n[k]

    logLikelihood = getLogLikelihood(means, weights, covariances, X)
 
    return weights, means, covariances, logLikelihood
