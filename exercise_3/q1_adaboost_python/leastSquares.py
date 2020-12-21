import numpy as np

def leastSquares(data, label):
    # Sum of squared error shoud be minimized
    #
    # INPUT:
    # data        : Training inputs  (num_samples x dim)
    # label       : Training targets (num_samples x 1)
    #
    # OUTPUT:
    # weights     : weights   (dim x 1)
    # bias        : bias term (scalar)
    #

    # Extend each datapoint x as [1, x]
    # (Trick to avoid modeling the bias term explicitly)
    num_samples = len(data)
    data = np.append(np.ones(num_samples).reshape(num_samples,1), data, axis = 1)  # before: (38x2), now: (38x3)

    # Take the pseudo inverse
    weight = (np.linalg.lstsq(data.T.dot(data),data.T)[0].dot(label)) # inv(A)*b = A\b shape: (3, 1)

    # Form the output
    bias = weight[0]  # get bias
    weight = weight[1:]  # get weights

    return [weight, bias]