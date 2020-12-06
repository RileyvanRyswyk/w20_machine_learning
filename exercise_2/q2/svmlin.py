import numpy as np
# might need to add path to mingw-w64/bin for cvxopt to work
# import os
# os.environ["PATH"] += os.pathsep + ...
import cvxopt


def svmlin(X, t, C):
    # Linear SVM Classifier
    #
    # INPUT:
    # X        : the dataset                  (num_samples x dim)
    # t        : labeling                     (num_samples x 1)
    # C        : penalty factor for slack variables (scalar)
    #
    # OUTPUT:
    # alpha    : output of quadprog function  (num_samples x 1)
    # sv       : support vectors (boolean)    (1 x num_samples)
    # w        : parameters of the classifier (1 x dim)
    # b        : bias of the classifier       (scalar)
    # result   : result of classification     (1 x num_samples)
    # slack    : points inside the margin (boolean)   (1 x num_samples)

    # Number of data points
    epsilon = 1e-6
    n, d = X.shape[0], X.shape[1]
    H = np.zeros((n, n))

    # H(n, m) = t_n * t_m * x_m^T x x_n
    for i in range(0, n):
        for j in range(0, n):
            H[i][j] = t[i] * t[j] * np.inner(X[j, :], X[i, :])
    
    H = cvxopt.matrix(H, (n, n))
    # Multiplier for a_n
    q = cvxopt.matrix(-1*np.ones(n), (n, 1))

    # Bound a's between a lower and upper Bound
    LB, UB = np.zeros(n), C*np.ones(n)
    G = cvxopt.matrix(np.vstack([-np.eye(n), np.eye(n)]), (2*n, n))
    h = cvxopt.matrix(np.hstack([-LB, UB]), (2*n, 1))

    # constraint
    A = cvxopt.matrix(t, (1, n))
    b = cvxopt.matrix(0.0)

    # solve it
    alpha = np.array(cvxopt.solvers.qp(H, q, G, h, A, b)['x'])[:,0]

    # boolean array indicating if point is a support vector
    sv = ((alpha >= epsilon) & (alpha <= C))

    # W is linear combination of a_n * t_n * x_n
    w = np.zeros(d)
    for i in range(0, n):
        w += alpha[i] * t[i] * X[i,:]

    # calculate offset, average over all suport vectors
    b = 0
    for i in range(0, n):
        if sv[i]:
            b += t[i]
            for j in range(0, n):
                if sv[j]:
                    b -= alpha[j] * t[j] * np.inner(X[j], X[i])
    b /= np.sum(sv)

    # classify the points 
    result = X @ w.T + b*np.ones(n)

    # points that are in the margin. 
    slack = sv & (np.absolute(t - result) > epsilon)
   
    return alpha, sv, w, b, result, slack
