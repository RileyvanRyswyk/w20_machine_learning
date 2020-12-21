import numpy as np
from numpy.random import choice
from leastSquares import leastSquares

def adaboostLSLC(X, Y, K, nSamples):
    # Adaboost with least squares linear classifier as weak classifier
    # for a D-dim dataset
    #
    # INPUT:
    # X         : the dataset (numSamples x numDim)
    # Y         : labeling    (numSamples x 1)
    # K         : number of weak classifiers (iteration number of Adaboost) (scalar)
    # nSamples  : number of data which are weighted sampled (scalar)
    #
    # OUTPUT:
    # alphaK    : voting weights (K x 1)
    # para      : parameters of least square classifier (K x 3) 
    #             For a D-dim dataset each least square classifier has D+1 parameters
    #             w0, w1, w2........wD

    #####Insert your code here for subtask 1e#####

    return [alphaK, para]
