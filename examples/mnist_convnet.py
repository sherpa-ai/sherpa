'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import os
import socket
# Before importing keras, decide which gpu to use.
try:
    # gpu_lock module located at /home/pjsadows/libs
    import gpu_lock

    GPUIDX = gpu_lock.obtain_lock_id()  # Return gpuid, or -1 if there was a problem.
except:
    print('Could not import gpu_lock. Prepend /extra/pjsadows0/libs/shared/gpu_lock/ to PYTHONPATH.')
    GPUIDX = 0
assert GPUIDX >= 0, '\nNo gpu available.'
print('\nRunning from GPU %s' % str(GPUIDX))
# Carefully import backend.
if os.environ['KERAS_BACKEND'] == 'theano':
    os.environ['THEANO_FLAGS'] = "floatX=float32,device=cuda{},base_compiledir=~/.theano/{}_gpu{}".format(GPUIDX, socket.gethostname(), GPUIDX)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUIDX)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf

    CONFIG = tf.ConfigProto(device_count={'GPU': 1}, log_device_placement=False, allow_soft_placement=False)
    CONFIG.gpu_options.allow_growth = True  # Prevents tf from grabbing all gpu memory.
    sess = tf.Session(config=CONFIG)
    from keras import backend as K
    K.set_session(sess)

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from collections import defaultdict
import pickle as pkl

batch_size = 128
num_classes = 10
epochs = 12
img_rows, img_cols = 28, 28

def get_mnist():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test


def get_model(hp):
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)

    num_filters = hp['num_filters']
    filter_size = hp['filter_size']
    dropout = hp['dropout']
    activation = hp['activation']

    model = Sequential()
    model.add(Conv2D(num_filters//2, kernel_size=(filter_size, filter_size),
                     activation=activation,
                     input_shape=input_shape))
    model.add(Conv2D(num_filters, (filter_size, filter_size), activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def main(modelfile, historyfile, hp={}, epochs=1, verbose=2):
    """
    ---------------------------------------------------------------------------
    EDIT THIS METHOD
    ---------------------------------------------------------------------------
    This main function is called by Sherpa.
    Input:
        modelfile  = File containing model.
        historyfile= File containing dictionary of per-epoch results.
        hp         = Dictionary of hyperparameters.
        epochs     = Number of epochs to train this round.
        verbose    = Passed to keras.fit_generator.
    Output:
        No return value is given, but updates modelfile and historyfile.
    """
    if os.path.isfile(historyfile):
        # Resume training.
        assert os.path.isfile(modelfile)
        assert hp is None or len(hp) == 0
        model = keras.models.load_model(modelfile)
        with open(historyfile, 'rb') as f:
            history = pkl.load(f)
        initial_epoch = len(history['loss'])  # Assumes loss is list of length epochs.
    else:
        # Create new model.
        model = get_model(hp=hp)
        history = defaultdict(list)
        initial_epoch = 0

    print('Running with {}'.format(str(hp)))

    # Define dataset.
    x_train, y_train, x_test, y_test = get_mnist()

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=verbose,
              validation_data=(x_test, y_test),
              initial_epoch=initial_epoch)

    # Update history and save to file.
    partialh = model.history.history
    for k in partialh:
        history[k].extend(partialh[k])
    with open(historyfile, 'wb') as fid:
        pkl.dump(history, fid)
    # Save model file if we want to restart.
    model.save(modelfile)

    return