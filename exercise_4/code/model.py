import nn_modules
import numpy as np
import logging


class NNModel:
    """Model providing functionality for training and testing a generic stack of NNModules"""

    def __init__(self, module_list):
        logging.debug('Start building model...')
        self.modules = []
        # Buffer storing the gradients wrt. the module outputs
        # Used in the separate parameter gradient calculation step
        self.bprop_buffer = dict()

        # Build module stack for the given module list
        for (Module, args) in module_list:
            if not issubclass(Module, nn_modules.NNModule):
                raise Exception('Not a valid NNModule')

            nn_module = Module(**args)
            self.bprop_buffer[nn_module] = None  # Add bprop_buffer
            self.modules.append(nn_module)  # Add module

        logging.debug('Model build')

    def init_network(self):
        """Initialize all modules

        :return:
        """
        logging.debug('Initializing model ...')
        for nn_module in self.modules:
            nn_module.initialize_parameter()

        logging.debug('Model initialized')

    def fprop(self, input):
        """Run forward-propagation through the entire module stack

        :param input:
        :return:
        """
        logging.debug('Forward propagation through model')
        for nn_module in self.modules:
            input = nn_module.fprop(input)

        logging.debug('Forward propagation done')

        return input

    def bprop(self, v):
        """Backpropagation through the complete module stack

        :param v: Gradients wrt. to model output.
        :return: None
        """
        logging.debug('Back-propagation through model ...')
        for nn_module in reversed(self.modules):
            self.bprop_buffer[nn_module] = np.array(v)
            v = nn_module.bprop(v)

        logging.debug('Back-propagation through model done')

    def update_parameters(self, up_fun):
        """Run parameter update step for all model parameters

        :param up_fun: Update function to be applied to parameters and their gradients
        :return:
        """
        logging.debug('Attempting to update model parameters...')
        for nn_module in self.modules:
            # Compute accumulated parameter gradients based on the previously calculated
            # and cached gradients at the layer ouputs
            grad_paras = nn_module.get_grad_param(self.bprop_buffer[nn_module])
            # If there are parameters to update
            if grad_paras:
                nn_module.apply_parameter_update(grad_paras, up_fun)

    def initialize_parameter(self):
        """Run initialization for all the modules

        :return:
        """
        for nn_module in self.modules:
            nn_module.initialize_parameter()
