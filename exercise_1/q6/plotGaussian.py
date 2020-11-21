import numpy as np
import matplotlib.pyplot as plt


def plotGaussian(mu, sigma):
    dimension = mu.shape[0]
    if len(mu.shape) > 1:
        n_components = mu.shape[1]
    else:
        n_components = 1
    plt.subplot()
    if dimension == 2:
        if n_components == 1 and sigma.shape == (2, 2):
            n = 36
            phi = np.arange(0, n, 1) / (n-1) * 2 * np.pi
            epoints = np.sqrt(np.abs(sigma)).dot([np.cos(phi), np.sin(phi)]) + mu[:, np.newaxis]
            plt.plot(epoints[0, :], epoints[1, :], 'r')
        else:
            print('ERROR: size mismatch in mu or sigma\n')
    else:
        raise ValueError('Only dimension 2 is implemented.')
