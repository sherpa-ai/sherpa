from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sherpa
import keras
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.optimizers import SGD
from keras.datasets import mnist


def define_model(params):
    """
    Return compiled model using hyperparameters specified in args.
    """
    nin = 784
    nout = 10
    act = params.get('act', 'relu')
    init = 'glorot_normal'
    arch = params.get('arch', [100, 100])
    dropout = params.get('dropout')
    input = Input(shape=(nin,), dtype='float32', name='input')
    x = input
    for units in arch:
        x = Dense(units, kernel_initializer=init, activation=act)(x)
        if dropout:
            x = Dropout(dropout)(x)
    output = Dense(nout, kernel_initializer=init, activation='softmax', name='output')(x)
    model = Model(inputs=input, outputs=output)

    # Learning Algorithm
    lrinit = params.get('lrinit', 0.02)
    momentum = params.get('momentum', 0.7)
    lrdecay = params.get('lrdecay', 0.)
    loss = {'output':'categorical_crossentropy'}
    metrics = {'output':'accuracy'}
    loss_weights = {'output':1.0}
    optimizer = SGD(lr=lrinit, momentum=momentum, decay=lrdecay)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights)
    return model


def main(client, trial):
    batch_size = 32
    num_classes = 10
    epochs = trial.parameters.get('epochs', 15)

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
    model = define_model(trial.parameters)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              callbacks=[client.keras_send_metrics(trial,
                                                   objective_name='val_loss',
                                                   context_names=['val_acc'])],
              validation_data=(x_test, y_test))


if __name__=='__main__':
    client = sherpa.Client()
    trial = client.get_trial()
    main(client, trial)

