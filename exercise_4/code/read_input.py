from numpy import genfromtxt
import logging


def read_entire_dataset(path_data, path_label):
    logging.info('Reading data...')
    data = genfromtxt(path_data, delimiter=' ', dtype=float)
    data /= 255.0
    labels = genfromtxt(path_label, delimiter=' ', dtype=int)
    logging.info('Data read')

    return data, labels
