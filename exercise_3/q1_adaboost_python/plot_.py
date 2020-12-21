import matplotlib.pyplot as plt
import numpy as np

def plot_(X, Y, j, theta, text):

    # Define some helper vars for plotting
    number = len(Y)
    xmin = -5 # min(X[:, 0])
    xmax = 5 # max(X[:, 0])
    ymin = -5 # min(X[:, 1])
    ymax = 5 # max(X[:, 1])
    
    # Plot the classifier together with the data
    plt.subplot()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(text)
    plt.axis([xmin, xmax, ymin, ymax])

    plt.plot(X[:, 0:1][Y == 1], X[:, 1:2][Y == 1], c='b', marker='x', linestyle='none', markersize=5)
    plt.plot(X[:, 0:1][Y == -1], X[:, 1:2][Y == -1], c='r', marker='o', linestyle='none', markersize=5,
             fillstyle='none')

    if j == 1:  # plot
        a = np.arange(ymin, ymax, (ymax-ymin)/number)
        plt.plot(np.ones(len(a))*theta, a, 'r-')
    else:
        a = np.arange(xmin, xmax, (xmax-xmin)/number)
        plt.plot(a, np.ones(len(a))*theta, 'r-')
