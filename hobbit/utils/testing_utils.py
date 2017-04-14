from __future__ import division
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.initializers import glorot_uniform, Zeros
from subprocess import check_output
import numpy as np
import re
import h5py
import os

def create_model(hparams):
    """
    Returns:
        A keras model for tests
    """
    # set seed for recreatability
    model = Sequential()
    model.add(Dense(hparams['num_units'], activation='relu', input_shape=(784,),
                      kernel_initializer=glorot_uniform(seed=1234), bias_initializer=Zeros()))
    model.add(Dense(10, activation='softmax',
                      kernel_initializer=glorot_uniform(seed=1234), bias_initializer=Zeros()))
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=hparams['lr']),
                  metrics=['accuracy'])
    return model


def create_model_two(hparams):
    """
    Returns:
        A keras model for tests
    """
    # set seed for recreatability
    model = Sequential()
    model.add(Dropout(rate=hparams['dropout'], input_shape=(784,)))
    model.add(Dense(100, activation=hparams['activation']))
    model.add(Dropout(rate=hparams['dropout']))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=hparams['learning_rate']),
                  metrics=['accuracy'])
    return model


def load_dataset(short=False):
    """
    Returns:
        A ready-to-train-on dataset for testing. Here, MNIST.
    """
    num_classes = 10

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)
    if short:
        return (x_train[:1000], y_train[:1000]), (x_test[:1000], y_test[:1000])
    else:
        return (x_train, y_train), (x_test, y_test)


def store_mnist_hdf5(dir):
    """
    Loads mnist and stores it as hdf5 in given folder
    """
    (x_train, y_train), (x_test, y_test) = load_dataset()

    hdf5_path = os.path.join(dir, 'mnist.h5')

    with h5py.File(hdf5_path) as f:
        f.create_dataset(name='x_train', shape=x_train.shape, dtype=x_train.dtype, data=x_train)
        f.create_dataset(name='y_train', shape=y_train.shape, dtype=y_train.dtype, data=y_train)
        f.create_dataset(name='x_test', shape=x_test.shape, dtype=x_test.dtype, data=x_test)
        f.create_dataset(name='y_test', shape=y_test.shape, dtype=y_test.dtype, data=y_test)

    return hdf5_path


def get_hdf5_generator(x, y, batch_size=100):
    """
    # Template generator
    x: any x array (hdf5 or numpy)
    y: y array
    """
    num_samples = y.shape[0]
    num_batches = np.ceil(num_samples/batch_size).astype('int')
    while True:
        for i in range(num_batches):
            v1 = i*batch_size
            v2 = min((i+1)*batch_size, num_samples)
            yield x[v1:v2], y[v1:v2]


def read_nvidia_smi(gpus=list(range(4)), cutoff=60):
    """
    Args:
        gpus: GPUs to be checked
        cutoff: Usage (in MiB) above which a gpu is declared to be in use

    Returns:
        True/False depending on whether those gpus are in use
    """

    s = check_output(["nvidia-smi"])
    matches = re.findall(r'([0-9]+)MiB / [0-9]+MiB', s)
    usage = map(int, matches)
    return all(gpu_usage >= cutoff for gpu_id, gpu_usage in enumerate(usage) if gpu_id in gpus)


def gpu_exists():
    """
    Returns:
        True/False depending on whether there is at least one GPU installed
    """
    try:
        check_output(["nvidia-smi"])
        return True
    except(OSError):
        return False