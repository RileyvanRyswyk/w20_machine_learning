import argparse
import read_input
import nn_modules
import model
import train
import logging
import pickle
import test


def train_model_task2(path_data, path_label):
    data, label = read_input.read_entire_dataset(path_data, path_label)

    # Module description
    module_desc = [(nn_modules.Linear, {'n_in': 28 * 28, 'n_out': 10}),
                   (nn_modules.Softmax, {})]

    # Construct model with given components/modules
    my_model = model.NNModel(module_desc)

    # Get loss
    loss = nn_modules.CrossEntropyLoss()

    train.train(my_model, loss, data, label, nbr_epochs=100, batch_size=600, lr=0.1)

    return my_model


def train_model_task3(path_data, path_label):
    data, label = read_input.read_entire_dataset(path_data, path_label)

    module_desc = [(nn_modules.Linear, {'n_in': 28 * 28, 'n_out': 200}),
                   (nn_modules.Tanh, {}),
                   (nn_modules.Linear, {'n_in': 200, 'n_out': 10}),
                   (nn_modules.Softmax, {})]

    my_model = model.NNModel(module_desc)

    loss = nn_modules.CrossEntropyLoss()

    train.train(my_model, loss, data, label, 100, 600, 0.1)

    return my_model


def train_model_task4(path_data, path_label):
    data, label = read_input.read_entire_dataset(path_data, path_label)

    module_desc = [(nn_modules.Linear, {'n_in': 28 * 28, 'n_out': 200}),
                   (nn_modules.Tanh, {}),
                   (nn_modules.Linear, {'n_in': 200, 'n_out': 10}),
                   (nn_modules.LogSoftmax, {})]

    my_model = model.NNModel(module_desc)

    loss = nn_modules.LogCrossEntropyLoss()

    train.train(my_model, loss, data, label, 100, 600, 0.1)

    return my_model

def train_model_task5(path_data, path_label):
    data, label = read_input.read_entire_dataset(path_data, path_label)

    module_desc = [(nn_modules.Linear, {'n_in': 3 * 32 * 32, 'n_out': 200}),
                   (nn_modules.Tanh, {}),
                   (nn_modules.Linear, {'n_in': 200, 'n_out': 100}),
                   (nn_modules.LogSoftmax, {})]

    my_model = model.NNModel(module_desc)

    loss = nn_modules.LogCrossEntropyLoss()

    train.train(my_model, loss, data, label, 100, 600, 0.1)

    return my_model


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--model_file",    type=str,   default="./model.pkl", help="Pickle file for model (dir has to exist)")
    parser.add_argument("--train",         type="bool", nargs="?", const=True, default=False, help="Run training")
    parser.add_argument("--test",          type="bool", nargs="?", const=True, default=False, help="Run test")
    parser.add_argument("--data_train",    type=str,   default='./data/mnist-train-data.csv', help="Training data file")
    parser.add_argument("--labels_train",  type=str,   default='./data/mnist-train-labels.csv', help="Training labels file")
    parser.add_argument("--data_test",     type=str,   default='./data/mnist-test-data.csv', help="Test data file")
    parser.add_argument("--labels_test",   type=str,   default='./data/mnist-test-labels.csv', help="Test labels file")
    parser.add_argument("--data_train2", type=str, default='./data/cifar100-train-data.csv', help="Training data file")
    parser.add_argument("--labels_train2", type=str, default='./data/cifar100-train-fine-labels.csv', help="Training labels file")
    parser.add_argument("--data_test2", type=str, default='./data/cifar100-test-data.csv', help="Test data file")
    parser.add_argument("--labels_test2", type=str, default='./data/cifar100-test-fine-labels.csv', help="Test labels file")
    parser.add_argument("--task",          type=int,   nargs=1, default=2, help="Task number of the sheet")

    FLAGS, unparsed = parser.parse_known_args()

    # Define/train model
    my_model = None
    if FLAGS.train:  # should we train the model?
        task = FLAGS.task[0]
        if task == 2:
            my_model = train_model_task2(FLAGS.data_train, FLAGS.labels_train)
        elif task == 3:
            my_model = train_model_task3(FLAGS.data_train, FLAGS.labels_train)
        elif task == 4:
            my_model = train_model_task4(FLAGS.data_train, FLAGS.labels_train)
        elif task == 5:
            my_model = train_model_task5(FLAGS.data_train2, FLAGS.labels_train2)

    # Save the model
    if my_model and FLAGS.model_file:
        logging.info('Saving model to file %s' % FLAGS.model_file)
        with open(FLAGS.model_file, 'wb') as f:
            pickle.dump(my_model, f)

    # Apply the model to test data
    if FLAGS.test:
        if not my_model:
            if FLAGS.model_file:
                logging.info('Reading model from file %s for testing...' % FLAGS.model_file)
                with open(FLAGS.model_file, 'rb') as f:
                    u = pickle._Unpickler(f)
                    u.encoding = 'latin1'
                    my_model = u.load()
            else:
                raise Exception("No model trained or given by file for testing")

        if task < 5:
            test.test_model(my_model, FLAGS.data_test, FLAGS.labels_test)
        elif task == 5:
            test.test_model(my_model, FLAGS.data_test2, FLAGS.labels_test2)


