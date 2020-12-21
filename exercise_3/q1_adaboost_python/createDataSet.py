import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", ".*GUI is implemented.*")  # remove false depracation warning


def createDataSet(number, xmin, xmax, ymin, ymax):
    # Create arbitrary datasets for evaluations
    #
    # INPUT:
    # number                 : specify the number of points you want to create for each class
    # xmin, xmax, ymin, ymax : range for the dataset
    #
    # OUTPUT:
    # X			 : dataset (numSamples x numDim)
    # Y			 : labeling (numSamples x 1)
    #
    #
    # INSTRUCTIONS:
    # Use left mouse button to create data points. 
    # You will get a message if the 'number' of points of the first class are created.
    # As next step just press enter and continue with creating the data point of the second class
    #
    # You can save your dataset using np.savez(["../data/synthetic1", 'X, 'Y')
    # np.savez('../data/synthetic1', 'X', 'Y') so on


    print('Create samples for first class:')
    plt.ion()
    fig = plt.figure()
    plt.axis([xmin, xmax, ymin, ymax])
    x = []
    y = []
    x1 = np.ndarray((number, 2))
    x2 = np.ndarray((number, 2))
    for i in range(number):
        x1[i] = plt.ginput(1)[0]
        x.append(x1[i][0])
        y.append(x1[i][1])
        plt.scatter(x, y, c='b', marker=(8, 2, 0), linewidth=0.5)
        plt.pause(0.1)

    print('First class finished.')
    #input('Press Enter to create samples for second class:')

    xx = []
    yy = []
    for i in range(number):
        x2[i] = plt.ginput(1)[0]  # np.random.random();
        xx.append(x2[i][0])
        yy.append(x2[i][1])
        plt.scatter(xx, yy, c='r', marker='+')
        plt.pause(0.1)
    plt.pause(0.1)
    plt.close(fig)
    plt.ioff()

    X = np.concatenate((x1, x2), axis=0)
    y1 = np.ones(number)
    y2 = np.ones(number) * (-1)
    Y = np.concatenate((y1, y2), axis=0)[:, np.newaxis]

    #np.savez('synthetic', X, Y)
    return X, Y
