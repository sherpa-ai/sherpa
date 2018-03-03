from __future__ import print_function
import os
import numpy as np
import sherpa

import keras
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import SGD
from keras.datasets import mnist


def define_model(params):
    '''
    Return compiled model using hyperparameters specified in args.
    '''
    nin    = 784
    nout   = 10
    units  = 10
    nhlay  = 2
    act    = params['act']
    init   = 'glorot_normal'
    input  = Input(shape=(nin,), dtype='float32', name='input')
    x      = input
    for units in params['arch']:
        x  = Dense(units, kernel_initializer=init, activation=act)(x)
    output = Dense(nout, kernel_initializer=init, activation='linear', name='output')(x)
    model  = Model(inputs=input, outputs=output)

    # Learning Algorithm
    lrinit    = params['lrinit']
    momentum  = params['momentum']
    lrdecay   = params['lrdecay']
    loss      = {'output':'categorical_crossentropy'}
    metrics   = {'output':'accuracy'}
    loss_weights = {'output':1.0}
    optimizer = SGD(lr=lrinit, momentum=momentum, decay=lrdecay)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights)
    return model


def main(client, trial):
    batch_size = 128
    num_classes = 10
    epochs = trial.parameters['epochs']

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    # Create new model.
    model   = define_model(trial.parameters)

    send_call = lambda epoch, logs: client.send_metrics(trial=trial,
                                                        iteration=epoch,
                                                        objective=logs['val_acc'],
                                                        context={'val_loss':
                                                                     logs[
                                                                         'val_loss']})
    callbacks = [keras.callbacks.LambdaCallback(on_epoch_end=send_call)]

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=(x_test, y_test))

    if 'modelfile' in trial.parameters:
        # Save model file.
        model.save(trial.parameters['modelfile'])

    return

if __name__=='__main__':
    client = sherpa.Client()
    try:
        trial = client.get_trial()
        main(client, trial)
    finally:
        gpu_lock.free_lock(GPUIDX)

