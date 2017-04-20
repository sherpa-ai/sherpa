from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from hobbit.algorithms import Hyperband
from hobbit import Hyperparameter

num_classes = 10

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# This will do preprocessing and realtime data augmentation:
train_gen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images
train_gen.fit(x_train)

# This will do preprocessing and realtime data augmentation:
valid_gen = ImageDataGenerator()  # by default all augmentation is off



def get_model(hparams):
    model = Sequential()

    model.add(Conv2D(hparams['num_layer_1_units'], (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(hparams['num_layer_1_units'], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(hparams['dropout_1']))

    model.add(Conv2D(2*hparams['num_layer_1_units'], (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(2*hparams['num_layer_1_units'], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(hparams['dropout_1']))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(hparams['dropout_2']))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


repo_dir = './cifar_repo'
hparam_ranges = [Hyperparameter(name='num_layer_1_units', distr_args=[np.arange(16,128,16)], distribution='choice'),
                 Hyperparameter(name='dropout_1', distr_args=[0., 0.8]),
                 Hyperparameter(name='dropout_2', distr_args=[0., 0.8])]
batch_size = 32


num_train_batches = np.ceil(x_train.shape[0]/batch_size).astype('int')
num_test_batches = np.ceil(x_test.shape[0] / batch_size).astype('int')

hband = Hyperband(model_function=get_model,
                  hparam_ranges=hparam_ranges,
                  repo_dir=repo_dir,
                  generator_function=(train_gen.flow, valid_gen.flow),
                  train_gen_args=(x_train, y_train, batch_size),
                  valid_gen_args=(x_test, y_test, batch_size),
                  steps_per_epoch=num_train_batches,
                  validation_steps=num_test_batches)

tab = hband.run(R=20, eta=3)