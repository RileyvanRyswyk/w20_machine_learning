import abc
import numpy as np


class NNModule:
    """ Class defining abstract interface every module has to implement

    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def fprop(self, input):
        """ Forwardpropagate the input through the module

        :param input: Input tensor for the module
        :return Output tensor after module application
        """
        return

    @abc.abstractmethod
    def bprop(self, grad_out):
        """ Backpropagate the gradient the output to the input

        :param grad_out: Gradients at the output of the module
        :return: Gradient wrt. input
        """
        return

    @abc.abstractmethod
    def get_grad_param(self, grad_out):
        """ Return gradients wrt. the parameters
        Calculate the gardients wrt. to the parameters of the module. Function already
        accumulates gradients over the batch -> Save memory and implementation issues using numpy avoid loops

        :param grad_out: Gradients at the output
        :return: Gradients wrt. the internal parameter accumulated over the batch
        """
        return

    @abc.abstractmethod
    def apply_parameter_update(self, acc_grad_para, up_fun):
        """ Apply the update function to the internal parameters.

        :param acc_grad_para: Accumulated gradients over the batch
        :param up_fun: Update function used
        :return:
        """
        return

    # If we would like to support different initialization techniques, we could
    # use an Initializer class
    # For simplicity use a fixed initialize for each module
    @abc.abstractmethod
    def initialize_parameter(self):
        """ Initialize the internal parameter

        :return:
        """


class NNModuleParaFree(NNModule):
    """Specialization of the NNModule for modules which do not have any internal parameters

    """
    __metaclass__ = abc.ABCMeta

    def initialize_parameter(self):
        # No initialization necessary
        return

    def get_grad_param(self, grad_out):
        # No parameter gradients
        return None

    def apply_parameter_update(self, acc_grad_para, up_fun):
        # No parameters to update
        return


class LossModule(NNModule):
    """Specialization of NNModule for losses which need target values

    """
    __metaclass__ = abc.ABCMeta

    def set_targets(self, t):
        """Saves expected targets.
        Does not copy the input.

        :param t: Expected target values.
        :return:
        """
        self.t = t

    def initialize_parameter(self):
        # No internal parameters
        return

    def get_grad_param(self, grad_out):
        # No gradient for internal parameter
        return None

    def apply_parameter_update(self, acc_grad_para, up_fun):
        # No update needed
        return


# Task 2 a)
class Linear(NNModule):
    """Module which implements a linear layer"""

    #####Insert your code here for subtask 2a#####
    def __init__(self, n_in, n_out):
        # Save input for bprop
        self.cache_in = None
        # data input dimension dim(x)
        self.n_in = n_in
        # data output dimension dim(y)
        self.n_out = n_out

    def fprop(self, input):
        # Save input for bprop
        self.cache_in = np.array(input)
        # compute W.T * x + b in a mini batch
        # input is {batch_size} x {n_in}
        batch_size = input.shape[0]
        # transpose and extend b {n_out x 1} to each batch {batch_size x n_out}
        b_ext = np.tile(self.b.T, (batch_size, 1))
        # compute y{batch_size x n_out} => input{batch_size x n_in} x W{n_in x n_out} + b{batch_size x n_out}
        return input @ self.W + b_ext

    def bprop(self, grad_out):
        # compute dE/dx as dE/dy * W to give
        # dE/dy{batch_size x n_out} x W.T{n_out x n_int} => dE/dx{batch_size x n_in}
        return grad_out @ self.W.T

    def get_grad_param(self, grad_out):
        params = {}
        # dW = cache.x * dE/dy
        params['dW'] = self.cache_in.T @ grad_out
        # db = col_sum(dE/dy)
        params['db'] = np.sum(grad_out, axis=0)
        return params

    def apply_parameter_update(self, acc_grad_para, up_fun):
        self.W = up_fun(para=self.W, grad_para=acc_grad_para['dW'])
        self.b = up_fun(para=self.b.T, grad_para=acc_grad_para['db']).T

    def initialize_parameter(self):
        # initialize W and b
        mean = np.sqrt(2 / (self.n_in + self.n_out))
        self.W = np.random.normal(0, mean, (self.n_in, self.n_out))
        self.b = np.zeros((self.n_out, 1))


# Task 2 b)
class Softmax(NNModuleParaFree):
    """Softmax layer"""

    #####Insert your code here for subtask 2b#####
    def __init__(self):
        # Save input for bprop
        self.cache_out = None

    def fprop(self, input):
        # softmax_i = exp(x_i) / sum(exp(x_j))
        # input {batch_size x n_in}
        exp = np.exp(input)
        # normalize each row (by row sum)
        softmax = exp / np.sum(exp, axis=1)[:, np.newaxis]
        # Save output for bprop
        self.cache_out = softmax
        return softmax

    def bprop(self, grad_out):
        # calculate each sample in the batch separately
        # sigma = y {batch_size x n_in}
        sigma = self.cache_out
        z = np.zeros_like(grad_out)
        for j in range(grad_out.shape[0]):
            # as per 1b) z_i = sigma_i * [v_i - v*sigma.T]
            # {1 x 1} =
            v_dot_sigma = grad_out[j, :] @ sigma[j, :]
            z_j = sigma[j, :] * (grad_out[j, :] - v_dot_sigma)
            z[j, :] = z_j
        return z



# Task 2 c)
class CrossEntropyLoss(LossModule):
    """Cross-Entropy-Loss-Module"""
    def __init__(self):
        # Save input for bprop
        self.cache_in = None

    def fprop(self, input):
        self.cache_in = np.array(input)
        sz_batch = input.shape[0]
        loss = -1 * np.log(input[np.arange(sz_batch), self.t])
        return loss

    def bprop(self, grad_out):
        sz_batch, n_in = self.cache_in.shape
        z = np.zeros((sz_batch, n_in))
        z[np.arange(sz_batch), self.t] =  \
            -1 * 1.0/self.cache_in[np.arange(sz_batch), self.t]
        np.multiply(grad_out, z, z)
        return z


# Task 3 b)
class Tanh(NNModuleParaFree):
    """Module implementing a Tanh acitivation function"""

    def __init__(self):
        # Cache output for bprop
        self.cache_out = None

    def fprop(self, input):
        output = np.tanh(input)
        self.cache_out = np.array(output)
        return output

    def bprop(self, grad_out):
        return np.multiply(grad_out, 1 - self.cache_out ** 2)


# Task 4 e)
class LogCrossEntropyLoss(LossModule):
    """Log-Cross-Entropy-Loss"""
    def __init__(self):
        self.sz_batch = self.n_in = None

    def fprop(self, input):
        self.sz_batch, self.n_in = input.shape
        loss = -1 * input[np.arange(self.sz_batch), self.t]
        return loss

    def bprop(self, grad_out):
        z = np.zeros((self.sz_batch, self.n_in))
        z[np.arange(self.sz_batch), self.t] = -1
        np.multiply(grad_out, z, z)
        return z


# Task 4 e)
class LogSoftmax(NNModuleParaFree):
    """Log-Softmax-Module"""

    def __init__(self):
        # Save output for bprop
        self.cache_out = None

    def fprop(self, input):
        # See 4a for stability reasons
        inp_max = np.max(input, 1)
        # Transpose for numpy broadcasting -> Subtract each batch max from the batch
        input = (input.T - inp_max).T
        exponentials = np.exp(input)
        log_normalization = np.log(np.sum(exponentials, 1))

        # Transpose -> Subtract log normalization for each batch and reshape to batch \times output
        output = (input.T - log_normalization).T
        self.cache_out = np.array(output)

        return output

    def bprop(self, grad_out):
        sz_batch, n_in = grad_out.shape
        sum_grad = np.sum(grad_out, 1).reshape((sz_batch, 1))
        sigma = np.exp(self.cache_out)
        z = grad_out - np.multiply(sum_grad, sigma)
        return z
