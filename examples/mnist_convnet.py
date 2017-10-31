'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import os
import argparse
import sherpa

os.environ['KERAS_BACKEND'] = 'tensorflow'

if True:
    # Before importing keras, decide which gpu to use.
    try:
        # gpu_lock module located at /home/pjsadows/libs
        import gpu_lock
        GPUIDX = gpu_lock.obtain_lock_id() # Return gpuid, or -1 if there was a problem.
    except:
        print('Could not import gpu_lock. Prepend /extra/pjsadows0/libs/shared/gpu_lock/ to PYTHONPATH.')
        GPUIDX = 0
    assert GPUIDX >= 0, '\nNo gpu available.'
    print('Running from GPU %s' % str(GPUIDX))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUIDX)
else:
    print('Running on CPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import *


parser = argparse.ArgumentParser()
parser.add_argument('--num_filters', type=int, default=16)
parser.add_argument('--filter_size', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--activation', type=str, default='relu')
# Args used by scheduler.
parser.add_argument('--index', type=int, default=0)
parser.add_argument('--metricsfile', type=str, default='./test.pkl')
parser.add_argument('--modelfile', type=str, default='./testmodelfile')
FLAGS = parser.parse_args()
HP = vars(FLAGS)

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

    num_filters = hp.get('num_filters', 32)
    filter_size = hp.get('filter_size', 3)
    dropout = hp.get('dropout', 0.)
    activation = hp.get('activation', 'relu')
    lr = hp.get('lr', 0.01)

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
                  optimizer=keras.optimizers.SGD(lr=lr, momentum=0.9),
                  metrics=['accuracy'])
    return model


def train_mnist():
    model = get_model(hp=HP)

    print('Running with {}'.format(str(HP)))

    # Define dataset.
    x_train, y_train, x_test, y_test = get_mnist()
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=10,
              validation_data=(x_test, y_test))

    metrics = history.history
    sherpa.send_metrics(index=FLAGS.index, metrics=metrics, metricsfile=FLAGS.metricsfile)
    return


if __name__=='__main__':
    train_mnist()
