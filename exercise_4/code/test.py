import read_input
import logging
import numpy as np


def test_model(my_model, path_data, path_label):
    data, label = read_input.read_entire_dataset(path_data, path_label)
    batch_size = 100

    accurate_pred = 0
    for batch_num in range(0, int(np.ceil(data.shape[0] / float(batch_size)) + 0.1)):
        logging.info('Batch %d' % batch_num)
        batch_indices = np.arange(batch_num * batch_size, min((batch_num + 1) * batch_size, data.shape[0]))

        batch_data = data[batch_indices, :]
        batch_label = label[batch_indices]

        model_out = my_model.fprop(batch_data)
        predictions = np.argmax(model_out, 1)

        accurate_batch_pred = np.isclose(predictions, batch_label)
        accurate_pred += np.sum(accurate_batch_pred)

    print("Errors:", data.shape[0] - accurate_pred)
    print("Test accuracy %f" % ((float(accurate_pred) / data.shape[0])))

