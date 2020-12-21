import numpy as np
from numpy.random import choice
from simpleClassifier import simpleClassifier
from eval_adaBoost_simpleClassifier import eval_adaBoost_simpleClassifier

def adaboostCross(X, Y, K, nSamples, percent):
    # Adaboost with an additional cross validation routine
    #
    # INPUT:
    # X         : training examples (numSamples x numDims )
    # Y         : training lables (numSamples x 1)
    # K         : number of weak classifiers to select (scalar)
    #             (the _maximal_ iteration count - possibly abort earlier)
    # nSamples  : number of training examples which are selected in each round. (scalar)
    #             The sampling needs to be weighted!
    # percent   : percentage of the data set that is used as test data set (scalar)
    #
    # OUTPUT:
    # alphaK    : voting weights (K x 1)
    # para      : parameters of simple classifier (K x 2)
    # testX     : test dataset (numTestSamples x numDim)
    # testY     : test labels  (numTestSamples x 1)
    # error	    : error rate on validation set after each of the K iterations (K x 1)

    #####Insert your code here for subtask 1d#####
    # Randomly sample a percentage of the data as test data set
    return alphaK, para, testX, testY, error

