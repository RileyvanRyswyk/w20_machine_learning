import tensorflow as tf
import os
import glob
from functools import partial
from PIL import Image

# Mapping from class labels used in the file to our labels
_MAP_CS_TO_TR_LABEL = {24: 0, 25: 1, 26: 2}  # {person, rider, car} - see cityscapes dataset

IMAGE_SIZE = [64, 64]   # Size the image is scaled to
NUM_CLASSES = 3         # Nbr of classes we want to distinguish
NUM_EX_TRAIN = 37911    # Nbr examples in training set
FILTER_SIZE = 900       # Minimum pixel count required


def get_dataset_cs(path, num_epochs, batch_size):

    """Builds and return a tensorflow dataset
    :param path: Path of the png-files
    :param num_epochs: Dataset can be used for this number of epochs
    :param batch_size: Number of examples returned for each poll
    :return: Tensorflow dataset, tensorflow dataset only containing the names of file of sufficient size
    """

    # Fetch filenames and build initial dataset of file and labels
    file_list = glob.glob(os.path.join(path, '*.png'))  # list of filename strings
    assert len(file_list) > 0, "no files found in " + path
    labels = _labels_from_file_names(file_list)         # get list of labels corresponding to images
    dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(file_list), tf.convert_to_tensor(labels)))

    #####Insert your code here for subtask 1b#####
    dataset_filtered_names = dataset.filter(
        partial(tf_filter_size, size_threshold=FILTER_SIZE)
    )

    # Parse image from filename
    dataset = dataset_filtered_names.map(
        partial(_parse_function, im_size=IMAGE_SIZE),
        num_parallel_calls=12)
    
    #####Insert your code here for subtask 1d#####
    # Basic normalization in each channel [0..1]
    dataset = dataset.map(lambda x, y: (x / 255, y),
                          num_parallel_calls=tf.data.AUTOTUNE,
                          deterministic=False)

    # dataset = dataset.repeat(num_epochs)  # repeat this dataset __num_epochs__ times
    dataset = dataset.batch(batch_size)  # combine consecutive elements of this dataset into batches

    return dataset, dataset_filtered_names  # return dataset and corresponding names


def _labels_from_file_names(filenames):
    """ Generates labels based on filename
    :param filenames: List of file names
    :return: List of labels for list of files
    """
    labels = []
    for name in filenames:
        id_ = int(name[-9:-4])  # Get last part of filename which reveals label of image
        base_id = id_ if (id_ < 1000) else id_ // 1000
        labels.append(_MAP_CS_TO_TR_LABEL[base_id])
    return labels


def tf_filter_size(filename, label, size_threshold):
    return tf.py_function(_filter_size, [filename, size_threshold], tf.bool)


def _filter_size(filename, size_threshold):
    """Checks if image contains more pixels than specified threshold
    :param filename: Name of input image
    :param size_threshold: Number of pixel threshold
    :return: True if number of pixels inside image is larger then size_threshold, false otherwise
    """
    # Lazy opening should avoid reading the entire file
    im = Image.open(filename.numpy())
    w, h = im.size

    #####Insert your code here for subtask 1b#####
    # reject images below threshold size
    return w * h >= size_threshold.numpy()


def _parse_function(filename, label, im_size):
    """Parses an image and its label for tensorflow
    :param filename: Image to parse
    :param label: Label of the image
    :param im_size: Target image size
    :return: (image, label) pair in expected format and size
    """

    label = tf.cast(label, tf.int64)  # convert to int64
    image_string = tf.io.read_file(filename)  # read the file content
    image_decoded = tf.image.decode_png(image_string, channels=3)  # decode image_string from PNG

    #####Insert your code here for subtask 1c#####
    # Resize images to dimensions given in im_size
    image_resized = tf.image.resize(
        image_decoded, im_size, method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False,
        antialias=False
    )
    return image_resized, label


if __name__ == "__main__":
    dataset, dataset_filtered_names = get_dataset_cs("cityscapesExtractedValResized", 1, 2)
