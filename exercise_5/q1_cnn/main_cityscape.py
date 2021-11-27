import argparse
import logging
import time
import train
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import input_cs
import model
import datetime

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        config = tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


def test_model_file(model_file_name, data_test):

    cnn_model = tf.keras.models.load_model(model_file_name)

    data_iter = iter(data_test)

    cm = None
    for imgs, labels in data_iter:
        predictions = cnn_model.predict(imgs, verbose=1)
        predictions = tf.argmax(predictions, axis=1)

        cm_loop = tf.math.confusion_matrix(
            labels, predictions, num_classes=input_cs.NUM_CLASSES, dtype=tf.dtypes.int32,
            name=None
        )

        if cm is None:
            cm = cm_loop
        else:
            cm = tf.math.add(cm, cm_loop)

    print(cm)



def run_test(sess, logits, labels, handle, test_handle, test_initializer):
    #####Insert your code here for subtask k#####
    return
    # See the validation code snipped given on how to iterate over the test set


# def main():
#     start_time = time.time()
#     logging.basicConfig(level=logging.INFO)
#     parser = argparse.ArgumentParser()
#     parser.register("type", "bool", lambda v: v.lower() == "true")
#     parser.add_argument("--max_epochs", type=int, default=9, help="Number of epochs training is run.")
#     parser.add_argument("--batch_size", type=int, default=64, help="Batch size used during training.")
#     parser.add_argument("--learning_rate", type=float, default=0.05, help="Initial learning rate.")
#     parser.add_argument("--data_dir_tr", type=str, default="cityscapesExtractedResized", help="Directory of the training data.")
#     parser.add_argument("--data_dir_val", type=str, default="cityscapesExtractedValResized", help="Directory of the validation data.")
#     parser.add_argument("--data_dir_test", type=str, default="cityscapesExtractedTestResized", help="Directory of the test data.")
#     parser.add_argument("--debug", type="bool", nargs='?', const=True, default=False, help="Use debugger to track down bad values during training.")
#     parser.add_argument("--validate_every", type=int, nargs=1, default=2, help="Run validation every x epochs.")
#     parser.add_argument("--run_training", type="bool", nargs='?', const=True, default=False, help="Training loop is run")
#     parser.add_argument("--run_test", type="bool", nargs='?', const=True, default=False, help="Run testing (after training if training is demanded")
#     parser.add_argument("--save_model_name", type=str, default='models/'+datetime.datetime.now().strftime("%Y-%m-%d_%H:%M"), help="File the model is saved to")
#     parser.add_argument("--restore_model", type=str, default=None, help="File the model is restore from")
#     parser.add_argument("--train_log_dir", type=str, default='log/'+datetime.datetime.now().strftime("%Y-%m-%d_%H:%M"), help="Directory for training logs")
#     FLAGS, unparsed = parser.parse_known_args()
#     validate_every = FLAGS.validate_every if type(FLAGS.validate_every) is int else FLAGS.validate_every[0]
#
#     with tf.Session() as sess:
#
#         # Get datasets for training, validation and test
#         data_train, data_train_filtered_names = input_cs.get_dataset_cs(FLAGS.data_dir_tr,
#                                                                         FLAGS.max_epochs,
#                                                                         FLAGS.batch_size)
#         data_val, _ = input_cs.get_dataset_cs(FLAGS.data_dir_val, 1, 128)
#         data_test, _ = input_cs.get_dataset_cs(FLAGS.data_dir_test, 1, 128)
#
#         # Get iterators for training validation, and test
#         # Define handle for switching between dataset iterator
#         handle = tf.placeholder(dtype=tf.string, shape=[])
#         iterator = tf.data.Iterator.from_string_handle(handle,
#                                                                data_train.output_types,
#                                                                data_train.output_shapes)
#         next_element = iterator.get_next()
#         imgs, labels = next_element
#
#         training_iterator = data_train.make_one_shot_iterator()
#
#         validation_iterator = data_val.make_initializable_iterator()
#         validation_initializer = validation_iterator.initializer
#
#         test_iterator = data_test.make_initializable_iterator()
#         test_initializer = test_iterator.initializer
#
#         training_handle = sess.run(training_iterator.string_handle())
#         validation_handle = sess.run(validation_iterator.string_handle())
#         test_handle = sess.run(test_iterator.string_handle())
#
#         logits = model.build_model(imgs, input_cs.NUM_CLASSES)
#         tf.summary.histogram(logits.op.name + '/activations', logits)
#
#         # Start tensorflow debug session
#         if FLAGS.debug:
#             logging.info("Start debug session")
#             sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#             sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
#
#         if FLAGS.run_training:
#             train.train(sess, FLAGS.max_epochs, FLAGS.batch_size,
#                         validate_every, FLAGS.learning_rate, logits, labels, handle, training_handle,
#                         validation_handle, validation_initializer, model_file=None,
#                         save_name=FLAGS.save_model_name, train_log_dir=FLAGS.train_log_dir)
#
#         if FLAGS.run_test:
#             if FLAGS.run_training:
#                 run_test(sess, logits, labels, handle, test_handle, test_initializer)
#             else:
#                 if FLAGS.restore_model:
#                     test_model_file(sess, FLAGS.restore_model, logits, labels, handle, test_handle, test_initializer)
#                 else:
#                     logging.error("Cannot test: No model trained or specified for restoring for testing")
#         logging.info("Run time: %s" % (time.time() - start_time))


def main():
    start_time = time.time()
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--max_epochs", type=int, default=20, help="Number of epochs training is run.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size used during training.")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="Initial learning rate.")
    parser.add_argument("--data_dir_tr", type=str, default="cityscapesExtractedResized", help="Directory of the training data.")
    parser.add_argument("--data_dir_val", type=str, default="cityscapesExtractedValResized", help="Directory of the validation data.")
    parser.add_argument("--data_dir_test", type=str, default="cityscapesExtractedTestResized", help="Directory of the test data.")
    parser.add_argument("--debug", type="bool", nargs='?', const=True, default=False, help="Use debugger to track down bad values during training.")
    parser.add_argument("--validate_every", type=int, nargs=1, default=2, help="Run validation every x epochs.")
    parser.add_argument("--run_training", type="bool", nargs='?', const=True, default=False, help="Training loop is run")
    parser.add_argument("--run_test", type="bool", nargs='?', const=True, default=False, help="Run testing (after training if training is demanded")
    parser.add_argument("--save_model_name", type=str, default='models/'+datetime.datetime.now().strftime("%Y-%m-%d_%H.%M"), help="File the model is saved to")
    parser.add_argument("--restore_model", type=str, default=None, help="File the model is restore from")
    parser.add_argument("--train_log_dir", type=str, default='log/'+datetime.datetime.now().strftime("%Y-%m-%d_%H.%M"), help="Directory for training logs")
    FLAGS, unparsed = parser.parse_known_args()
    validate_every = FLAGS.validate_every if type(FLAGS.validate_every) is int else FLAGS.validate_every[0]

    # Get datasets for training, validation and test
    data_train, data_train_filtered_names = input_cs.get_dataset_cs(FLAGS.data_dir_tr,
                                                                    FLAGS.max_epochs,
                                                                    FLAGS.batch_size)

    data_val, _ = input_cs.get_dataset_cs(FLAGS.data_dir_val, 1, 128)
    data_test, _ = input_cs.get_dataset_cs(FLAGS.data_dir_test, 1, 128)

    # Get iterators for training validation, and test
    # Define handle for switching between dataset iterator
    # handle = tf.placeholder(dtype=tf.string, shape=[])
    # iterator = tf.data.Iterator.from_string_handle(handle,
    #                                                        data_train.output_types,
    #                                                        data_train.output_shapes)
    # train_iterator = iter(data_train)
    # next_element = iterator.get_next()
    # imgs, labels = next_element

    #
    # # Start tensorflow debug session
    # if FLAGS.debug:
    #     logging.info("Start debug session")
    #     sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #     sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    #
    if FLAGS.run_training:
        q1_model = model.build_model(input_cs.NUM_CLASSES)
        train.train(num_epochs=FLAGS.max_epochs,
                    batch_size=FLAGS.batch_size,
                    data_test=data_test,
                    initial_lr=FLAGS.learning_rate,
                    cnn_model=q1_model,
                    data_train=data_train,
                    data_val=data_val,
                    model_file=None,
                    save_name=FLAGS.save_model_name,
                    train_log_dir=FLAGS.train_log_dir)

    if FLAGS.run_test:
        test_model_file(FLAGS.restore_model, data_test)

if __name__ == "__main__":
    # run this script e.g. like this
    # python3 main_cityscape.py --run_training --run_test --train_log_dir log/ --save_model_name models/model1

    main()
