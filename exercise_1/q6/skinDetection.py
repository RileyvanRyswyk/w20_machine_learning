import numpy as np
from estGaussMixEM import estGaussMixEM
from calculateProbability import calculateProbability


def skinDetection(ndata, sdata, K, n_iter, epsilon, theta, img):
    # Skin Color detector
    #
    # INPUT:
    # ndata         : data for non-skin color
    # sdata         : data for skin-color
    # K             : number of modes
    # n_iter        : number of iterations
    # epsilon       : regularization parameter
    # theta         : threshold
    # img           : input image
    #
    # OUTPUT:
    # result        : Result of the detector for every image pixel

    # train models 
    s_weights, s_means, s_covariances = estGaussMixEM(sdata, K, n_iter, epsilon)
    n_weights, n_means, n_covariances = estGaussMixEM(ndata, K, n_iter, epsilon)

    result = []
    for i, img_row in enumerate(img):
      row = []
      for j, pixel in enumerate(img_row):
        p_skin = calculateProbability(s_means, s_weights, s_covariances, pixel)
        p_nskin = calculateProbability(n_means, n_weights, n_covariances, pixel)
        row.append((p_skin / p_nskin) > theta)
      result.append(row)

    return result
