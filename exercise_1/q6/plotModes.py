from plotGaussian import plotGaussian
import matplotlib.pyplot as plt


def plotModes(means, covMats, X):
    plt.subplot()
    plt.scatter(X[:, 0], X[:, 1])
    M = means.shape[1]

    for i in range(M):
        plotGaussian(means[:, i], covMats[:, :, i])
