import numpy as np
import math as math

def calculateProbability(means, weights, covariances, x):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # x              : Input data point D
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # probability  : probability

    #initialize
    probability = 0
    D = len(x)
    
    # calculate normalization factors for k gaussians 
    # and calculate Covariance inverses for k gaussians
    # we will need to reshape the covariance matrix to KxDxD from DxDxK
    covar_trans = covariances.transpose(2, 0, 1)
    norm_factors = 1 / (np.linalg.det(covar_trans) ** (1 / 2))
    norm_factors *= 1 / (2 * math.pi) ** (D / 2)
    inverses = np.linalg.inv(covar_trans)

    if type(means) != np.ndarray:
      means = np.array(means)

    if type(x) != np.ndarray:
      x = np.array(x)

    # iterate over each K gaussian 
    for k, gaussian_weight in enumerate(weights):

      # x - mu
      y = (x - means[k,:])

      # exp^{-1/2 * (x_n-u_k)^T * (covariance_k)^-1 * (x_n-u_k)}
      p = norm_factors[k] * math.exp(-0.5 * y.transpose() @ inverses[k,:,:] @ y)

      probability += gaussian_weight * p

    return probability

