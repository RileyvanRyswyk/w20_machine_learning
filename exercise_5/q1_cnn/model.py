import tensorflow as tf
import input_cs
import numpy as np


# def build_model(images, num_classes):
#     """Build the model
#     :param images: Input image placeholder
#     :param num_classes: Nbr of final output classes
#     :return: Output of final fc-layer
#     """
#     #####Insert your code here for subtask 1e#####
#     #####Insert your code here for subtask 1f#####
#     cnn_model = CNNModule(num_classes=num_classes, name='q1')
#     softmax_logits = cnn_model(images)
#
#     return softmax_logits

def build_model(num_classes):
    """Build the model
    :param num_classes: Nbr of final output classes
    :return: model
    """
    #####Insert your code here for subtask 1e#####
    #####Insert your code here for subtask 1f#####
    k_model = tf.keras.Sequential()
    k_model.add(tf.keras.layers.Conv2D(
        filters=24, kernel_size=5, strides=(1, 1), padding='same',
        data_format='channels_last', dilation_rate=(1, 1), groups=1, activation='relu',
        use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None,
        bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
        bias_constraint=None,
    ))
    k_model.add(tf.keras.layers.MaxPool2D(
        pool_size=(3, 3), strides=2, padding='same', data_format=None,
    ))
    k_model.add(tf.keras.layers.Conv2D(
        filters=32, kernel_size=5, strides=(1, 1), padding='same',
        data_format='channels_last', dilation_rate=(1, 1), groups=1, activation='relu',
        use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None,
        bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
        bias_constraint=None,
    ))
    k_model.add(tf.keras.layers.MaxPool2D(
        pool_size=(3, 3), strides=2, padding='same', data_format=None,
    ))
    k_model.add(tf.keras.layers.Conv2D(
        filters=50, kernel_size=5, strides=(1, 1), padding='same',
        data_format='channels_last', dilation_rate=(1, 1), groups=1, activation='relu',
        use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None,
        bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
        bias_constraint=None,
    ))
    k_model.add(tf.keras.layers.MaxPool2D(
        pool_size=(3, 3), strides=2, padding='same', data_format=None,
    ))
    k_model.add(tf.keras.layers.Flatten())
    k_model.add(tf.keras.layers.Dense(
        units=100, activation='relu', use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None,
        bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
        bias_constraint=None
    ))
    k_model.add(tf.keras.layers.Dense(
        units=50, activation='relu', use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None,
        bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
        bias_constraint=None
    ))
    k_model.add(tf.keras.layers.Dense(
        units=num_classes, activation='relu', use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None,
        bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
        bias_constraint=None
    ))

    k_model.compile(
        optimizer='sgd',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(),
            # tf.keras.metrics.Accuracy()
        ]
    )

    return k_model

