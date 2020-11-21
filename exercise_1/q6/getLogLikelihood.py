import numpy as np
import math as math

def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood

    #initialize
    logLikelihood = 0
    D = X.shape[1]
    
    # calculate normalization factors for k gaussians 
    # and calculate Covariance inverses for k gaussians
    # we will need to reshape the covariance matrix to KxDxD from DxDxK
    covar_trans = covariances.transpose(2, 0, 1)
    norm_factors = 1 / (np.linalg.det(covar_trans) ** (1 / 2))
    norm_factors *= 1 / (2 * math.pi) ** (D / 2)
    inverses = np.linalg.inv(covar_trans)

    if type(means) != np.ndarray:
      means = np.array(means)

    # iterate over the data points to determine their contributions
    for data_point in X:

      p_data_point = 0

      # iterate over each K gaussian 
      for k, gaussian_weight in enumerate(weights):

        # x - mu
        y = (data_point - means[k,:])

        # exp^{-1/2 * (x_n-u_k)^T * (covariance_k)^-1 * (x_n-u_k)}
        p = norm_factors[k] * math.exp(-0.5 * y.transpose() @ inverses[k,:,:] @ y)

        p_data_point += gaussian_weight * p
      logLikelihood += np.log(p_data_point)

    return logLikelihood

