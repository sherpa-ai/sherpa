# Train simple network on 2D data.
# Author: Peter Sadowski
from __future__ import print_function
import numpy as np
import sherpa
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import SGD

 
def dataset_bianchini(batchsize, k=1):
    '''
    Synthetic data set where we can control Betti numbers from Bianchini et al. 2014.
    Input: 2D real values, Output: binary {0,1}.
    f = g(t_k(x)), where g=1-||x||^2, t_1(x)=[1-2*x_1^2, 1-2*x_2^2], t_k = t * t_{k-1}
    '''
    g = lambda x: 1. - np.linalg.norm(x, ord=2)**2
    t = lambda x: 1. - 2.*(x**2)
    def f(x):
        for i in range(k):
            x = t(x)
        return g(x)
    while True:
        X = np.random.uniform(low=-1.,high=1.0 , size=(batchsize, 2))
        Y = (np.apply_along_axis(f, axis=1, arr=X) > 0.0).astype('float32')
        yield {'input':X}, {'output':Y}

def define_model(params):
    '''
    Return compiled model using hyperparameters specified in args.
    ''' 
    nin    = 2
    nout   = 1
    units  = 10
    nhlay  = 2
    act    = params['act']
    init   = 'glorot_normal'
    input  = Input(shape=(nin,), dtype='float32', name='input')
    x      = input
    for units in params['arch']:
        x  = Dense(units, kernel_initializer=init, activation=act)(x)
    output = Dense(nout, kernel_initializer=init, activation='sigmoid', name='output')(x)
    model  = Model(inputs=input, outputs=output)

    # Learning Algorithm
    lrinit    = params['lrinit']
    momentum  = params['momentum']
    lrdecay   = params['lrdecay']
    loss      = {'output':'binary_crossentropy'}
    metrics   = {'output':'accuracy'}
    loss_weights = {'output':1.0}
    optimizer = SGD(lr=lrinit, momentum=momentum, decay=lrdecay)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights)
    return model


def main(client, trial):
    # Create new model.
    model   = define_model(trial.parameters)

    # Define dataset.
    gtrain = dataset_bianchini(batchsize=100, k=3)
    gvalid = dataset_bianchini(batchsize=100, k=3)

    model.fit_generator(gtrain,
                        steps_per_epoch=100,
                        validation_data=gvalid,
                        validation_steps=10,
                        callbacks=[client.keras_send_metrics(trial, objective_name='val_loss', context_names=['val_acc'])],
                        epochs = trial.parameters['epochs'],
                        verbose=2)


if __name__ == '__main__':
    client = sherpa.Client()
    trial = client.get_trial()
    main(client, trial)

