import numpy as np
from functools import partial
import logging


def train(model, loss, data, label, nbr_epochs, batch_size, lr):
    """ Basic training routine

    :param model: Model to be trained
    :param loss: Loss
    :param data: Training data
    :param label: Training labels
    :param nbr_epochs: Nbr of epochs
    :param batch_size: Batch size
    :param lr: Learning rate
    :return:
    """

    # Initialization
    model.initialize_parameter()

    # Use standard gradient descent for updating the parameters
    up_fun = partial(vanilla_grad_desc, lr=lr)
    nbr_data, _ = data.shape

    # Run training epochs
    for e in range(0, nbr_epochs):
        logging.info('-------- Running epoch %d --------' % e)

        # Use random permutation of the data
        indices = np.random.permutation(np.arange(nbr_data))
        epoch_loss = 0

        # Iterate over all mini-batches from this epoch
        for batch_num in range(0, int(np.ceil(float(nbr_data) / batch_size))):
            logging.info('Batch %d' % batch_num)

            # Fetch data (randomized)
            batch_indices = indices[np.arange(batch_num * batch_size, min((batch_num + 1) * batch_size, nbr_data))]
            batch_data = data[batch_indices, :]
            batch_label = label[batch_indices]

            # Forward propagation
            model_out = model.fprop(batch_data)

            # Compute loss
            loss.set_targets(batch_label)
            batch_losses = loss.fprop(model_out)
            epoch_loss += np.mean(batch_losses)

            # Print loss
            logging.info(np.mean(batch_losses))

            # Back-propagate the loss averaged over batch
            sz_batch = len(batch_indices)
            z = loss.bprop(np.tile(1.0/sz_batch, (sz_batch, 1)))
            model.bprop(z)

            # Update model parameters implicitly using back-propagated gradients
            model.update_parameters(up_fun)

        epoch_loss /= int(np.ceil(float(nbr_data) / batch_size))
        logging.info("epoch %s %s", e, epoch_loss)

def vanilla_grad_desc(para, grad_para, lr):
    """ Update function for the vanilla gradient descent: w = w - learningRate * grad_w

    :param para: Parameter to be updated
    :param grad_para: Gradient at the parameter
    :param lr: learning rate
    :return:
    """
    return para - lr * grad_para
