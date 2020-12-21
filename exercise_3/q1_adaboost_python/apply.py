import numpy as np
import matplotlib.pyplot as plt
from simpleClassifier import simpleClassifier
from createDataSet import createDataSet
from scipy import io
from adaboostSimple import adaboostSimple
from eval_adaBoost_simpleClassifier import eval_adaBoost_simpleClassifier
from eval_adaBoost_leastSquare import eval_adaBoost_leastSquare
from adaboostCross import adaboostCross
from adaboostLSLC import adaboostLSLC
from adaboostUSPS import adaboostUSPS
from generate_grid import generate_grid
from plot_decision_surface import plot_decision_surface
from plot_logits import plot_logits
from plot_ import plot_

# Generate or load training data

# X, Y = createDataSet(3, -5, 5, -5, 5)  # Create your own dataset

syn = io.loadmat('synthetic')  # ... or load existing dataset
X = syn['X']
Y = syn['Y']


# b) Simple weak classifier training
j, theta = simpleClassifier(X, Y)
plt.subplot()
plot_(X, Y, j, theta, 'Simple weak linear classifier')
plt.show()

# c) Adaboost using the simple classifiers

kMax = 50          # Number of weak classifiers
nSamples = 20      # Number of random samples to train each classifier

# Compute parameters of K classifiers and the voting weight for each classifier
alphaK, para = adaboostSimple(X, Y, kMax, nSamples)

# Sample test data from a regular grid to illustrate the decision regions
X_grid, Y_grid, x, y = generate_grid(X)
X_ = np.vstack((X_grid.reshape((-1,)), Y_grid.reshape((-1,)))).T

# Compute discrete class predictions and continuous class probabilities
classLabels, logits = eval_adaBoost_simpleClassifier(X_, alphaK, para)

# Show decision surface
plot_decision_surface(X_, classLabels, X, Y, 'Decision Surface')

# Visualize logits
logits_r = np.reshape(logits, (len(x), len(y)))  # reshape into original shape
plot_logits(logits_r, 'Weighted sum of weak classifier results')


# d) Adaboost with cross-validation

kMax = 50          # Number of weak classifiers
nSamples = 20      # Number of random samples to train each classifier
percent = 0.2      # Percentage of test data

alphaK, para, testX, testY, error = adaboostCross(X, Y, kMax, nSamples, percent)

# Plot the classification error
plt.subplot()
plt.plot(error, linewidth = 3)
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.title('Classification Error')
plt.show()

# Sample data from a regular grid to illustrate the decision regions
X_grid, Y_grid, x, y = generate_grid(X)
X_ = np.vstack((X_grid.ravel(), Y_grid.ravel())).T

# Apply classifier to grid-sampled data
classLabels, logits = eval_adaBoost_simpleClassifier(X_, alphaK, para)

# Show decision surface
plot_decision_surface(X_, classLabels, X, Y, 'Decision Surface Cross-Validation')

# Visualize logits
logits_r = np.reshape(logits, (len(x), len(y)))  # reshape into original shape
plot_logits(logits_r, 'Weighted sum of weak classifier results')


# e) AdaBoost with least-square classifier compared to simple classifier

# What is the influence of these parameters?
kMax = 100         # Number of weak classifiers
nSamples = 20      # Number of random samples to train each classifier

# Train both classifiers
alphaK_ls, para_ls = adaboostLSLC(X, Y, kMax, nSamples)
alphaK_sc, para_sc = adaboostSimple(X, Y, kMax, nSamples)
#print(alphaK_ls, para_ls)
# Sample data from a regular grid to illustrate the decision regions
X_grid, Y_grid, x, y = generate_grid(X)
X_ = np.vstack((X_grid.ravel(), Y_grid.ravel())).T

# Apply both classifiers on grid data
classLabels_ls, logits_ls = eval_adaBoost_leastSquare(X_, alphaK_ls, para_ls)
classLabels_sc, logits_sc = eval_adaBoost_simpleClassifier(X_, alphaK_sc, para_sc)

# Plot least-square classifier
plot_decision_surface(X_, classLabels_ls, X, Y, 'Decision surface for least square classifier')
logits_ls_r = np.reshape(logits_ls, (len(x), len(y)))
plot_logits(logits_ls_r, 'Weighted sum of all leastsquare classifiers')

# Plot simple classifier
plot_decision_surface(X_, classLabels_sc, X, Y, 'Decision surface for simple classifier')
logits_sc_r = np.reshape(logits_sc, (len(x), len(y)))
plot_logits(logits_sc_r, 'Weighted sum of all simple classifiers')


# f) Least square based AdaBoost with USPS dataset

usps = io.loadmat('usps')   # Load USPS dataset
X = usps['X']
Y = usps['Y']
kMax = 50       # Number of weak classifiers
nSamples = 200  # Number of random samples to train each classifier
percent = 0.5   # Percentage of test data

iter = 3                    # How many times do we want to run it?
errors = np.ndarray((iter, kMax))  # Placeholder for errors
print('Running adaboostUSPS...')
for i in range(iter):            # Run it multiple times
    alphaK, para, error = adaboostUSPS(X, Y, kMax, nSamples, percent)
    errors[i, :] = error


# Plot error over iterations for multiple runs
plt.subplot()
for i in range(iter):
    plt.plot(np.arange(1, kMax+1), errors[i, :], linewidth=3)
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.title('Classification Error')
plt.show()
