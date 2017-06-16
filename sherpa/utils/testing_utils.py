from __future__ import division
import keras
import keras.backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.initializers import glorot_uniform, Zeros
import numpy as np
import h5py
import os

def create_model(hparams):
    """
    Returns:
        A keras model for tests
    """
    # set seed for recreatability
    model = Sequential()
    model.add(Dense(int(hparams['num_units']), activation='relu', input_shape=(
        784,),
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


def load_dataset(short=False, as_images=False):
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

    if as_images:
        img_rows, img_cols = 28, 28
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

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
            from_ = i*batch_size
            to_ = min((i+1)*batch_size, num_samples)
            yield x[from_:to_], y[from_:to_]





def branin(x1, x2):
    ##########################################################################
    #
    # BRANIN FUNCTION
    #
    # Authors: Sonja Surjanovic, Simon Fraser University
    #          Derek Bingham, Simon Fraser University
    # Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
    #
    # Copyright 2013. Derek Bingham, Simon Fraser University.
    #
    # THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY
    # FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
    # derivative works, such modified software should be clearly marked.
    # Additionally, this program is free software; you can redistribute it
    # and/or modify it under the terms of the GNU General Public License as
    # published by the Free Software Foundation; version 2.0 of the License.
    # Accordingly, this program is distributed in the hope that it will be
    # useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    # of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    # General Public License for more details.
    #
    # For function details and reference information, see:
    # http://www.sfu.ca/~ssurjano/
    #
    ##########################################################################
    #
    # INPUTS:
    #
    # xx = c(x1, x2)
    # a = constant (optional), with default value 1
    # b = constant (optional), with default value 5.1/(4*pi^2)
    # c = constant (optional), with default value 5/pi
    # r = constant (optional), with default value 6
    # s = constant (optional), with default value 10
    # t = constant (optional), with default value 1/(8*pi)
    #
    ##########################################################################
    import numpy as np
    a=1.
    b=5.1/(4.*np.pi**2)
    c=5./np.pi
    r=6.
    s=10.
    t=1./(8.*np.pi)

    term1 = a * (x2 - b*x1**2 + c*x1 - r)**2
    term2 = s*(1-t)*np.cos(x1)

    y = term1 + term2 + s
    return y