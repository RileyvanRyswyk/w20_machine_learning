import numpy as np
from numpy.random import choice
from leastSquares import leastSquares
from eval_adaBoost_leastSquare import eval_adaBoost_leastSquare


def adaboostUSPS(X, Y, K, nSamples, percent):
    # Adaboost with least squares linear classifier as weak classifier on USPS data
    # for a high dimensional dataset
    #
    # INPUT:
    # X         : the dataset (numSamples x numDim)
    # Y         : labeling    (numSamples x 1)
    # K         : number of weak classifiers (scalar)
    # nSamples  : number of data points obtained by weighted sampling (scalar)
    #
    # OUTPUT:
    # alphaK    : voting weights (1 x k) 
    # para      : parameters of simple classifier (K x (D+1))            
    #             For a D-dim dataset each simple classifier has D+1 parameters
    # error     : training error (1 x k)

    #####Insert your code here for subtask 1f#####
    # Sample random a percentage of data as test data set
    return [alphaK, para, error]
