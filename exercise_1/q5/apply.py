import numpy as np
from parameters import parameters
from knn import knn
from kde import kde
from gauss1D import gauss1D
import matplotlib.pyplot as plt

h, k = parameters()

print('Question: Kernel/K-Nearest Neighborhood Density Estimators')

# Produce the random samples
samples = np.random.normal(0, 1, 100)

# Compute the original normal distribution
realDensity = gauss1D(0, 1, 100, 5)

# Estimate the probability density using the KDE
estDensity = kde(samples, h)

# plot results
plt.subplot(2, 1, 1)
plt.plot(estDensity[:, 0], estDensity[:, 1], 'r', linewidth=1.5, label='KDE Estimated Distribution')
plt.plot(realDensity[:, 0], realDensity[:, 1], 'b', linewidth=1.5, label='Real Distribution')
plt.legend()

# Estimate the probability density using KNN
estDensity = knn(samples, k)

# Plot the distributions
plt.subplot(2, 1, 2)
plt.plot(estDensity[:, 0], estDensity[:, 1], 'r', linewidth=1.5, label='KNN Estimated Distribution')
plt.plot(realDensity[:, 0], realDensity[:, 1], 'b', linewidth=1.5, label='Real Distribution')
plt.legend()
plt.show()
