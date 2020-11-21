import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from EStep import EStep
from MStep import MStep
from regularize_cov import regularize_cov


def estGaussMixEM(data, K, n_iters, epsilon):
    # EM algorithm for estimation gaussian mixture mode
    #
    # INPUT:
    # data           : input data, N observations, D dimensional
    # K              : number of mixture components (modes)
    #
    # OUTPUT:
    # weights        : mixture weights - P(j) from lecture
    # means          : means of gaussians
    # covariances    : covariancesariance matrices of gaussians
    # logLikelihood  : log-likelihood of the data given the model

    # initialize parameters
    D = data.shape[1]

    # initial weights are uniform 
    weights = np.ones(K) / K    

    # Find initial means using K-Means
    kmeans = KMeans(n_clusters = K, n_init = 10).fit(data)
    cluster_idx = kmeans.labels_
    means = kmeans.cluster_centers_

    # Create initial covariance matrices
    covariances = np.zeros((D, D, K))
    for j in range(K):
      data_cluster = data[cluster_idx == j]
      min_dist = np.inf
      for i in range(K):
        # compute sum of distances in cluster
        dist = np.mean(euclidean_distances(data_cluster, [means[i]], squared=True))
        if dist < min_dist:
          min_dist = dist
      covariances[:, :, j] = np.eye(D) * min_dist

    # iteration counter and log likelihood array
    i, ll = 0, []
    while (i < n_iters) and (len(ll) < 2 or np.abs(ll[-1] - ll[-2]) > 0):
      for j in range(0, K):
        covariances[:,:,j] = regularize_cov(covariances[:,:,j], epsilon) 
      [logLikelihood, gamma] = EStep(means, covariances, weights, data)
      weights, means, covariances, logLikelihood = MStep(gamma, data)

      ll.append(logLikelihood)
      i += 1

    return [weights, means, covariances]
