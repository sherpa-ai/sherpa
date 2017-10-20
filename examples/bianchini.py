# Train simple network on 2D data.
# Author: Peter Sadowski
from __future__ import print_function
import sys, os
import numpy as np
import argparse
import ast

import glob
from collections import defaultdict

import sherpa
from sherpa.hyperparameters import DistributionHyperparameter as Hyperparameter

os.environ['KERAS_BACKEND'] = 'tensorflow'

if False:
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
    # Carefully import backend.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUIDX)
else:
    print('Running on CPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
CONFIG = tf.ConfigProto(device_count = {'GPU': 1}, log_device_placement=False, allow_soft_placement=False) 
CONFIG.gpu_options.allow_growth = True # Prevents tf from grabbing all gpu memory.
sess = tf.Session(config=CONFIG)
from keras import backend as K
K.set_session(sess)

import keras
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import SGD

#print(sys.argv)
parser = argparse.ArgumentParser()
parser.add_argument('--act', type=str)
parser.add_argument('--lrinit', type=float)
parser.add_argument('--momentum', type=float)
parser.add_argument('--lrdecay', type=float)
parser.add_argument('--arch', type=ast.literal_eval)
parser.add_argument('--epochs', type=int)
# Args used by scheduler.
parser.add_argument('--index', type=int)
parser.add_argument('--verbose', default=2, type=int) # Use verbose=2 for parallel keras jobs. 
parser.add_argument('--metricsfile', type=str)
parser.add_argument('--modelfile', type=str)
args = parser.parse_args()
#print(args) 
 
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

def define_model(args):
    '''
    Return compiled model using hyperparameters specified in args.
    ''' 
    nin    = 2
    nout   = 1
    units  = 10
    nhlay  = 2
    act    = args.act
    init   = 'glorot_normal'
    input  = Input(shape=(nin,), dtype='float32', name='input')
    x      = input
    for units in args.arch:
        x  = Dense(units, kernel_initializer=init, activation=act)(x)
    output = Dense(nout, kernel_initializer=init, activation='sigmoid', name='output')(x)
    model  = Model(inputs=input, outputs=output)

    # Learning Algorithm
    lrinit    = args.lrinit
    momentum  = args.momentum
    lrdecay   = args.lrdecay 
    loss      = {'output':'binary_crossentropy'}
    metrics   = {'output':'accuracy'}
    loss_weights = {'output':1.0}
    optimizer = SGD(lr=lrinit, momentum=momentum, decay=lrdecay)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights)
    return model

def main(args):
    # Create new model.
    model   = define_model(args)

    # Define dataset.
    gtrain = dataset_bianchini(batchsize=100, k=3)
    gvalid = dataset_bianchini(batchsize=100, k=3)
    
    # Train model. 
    initial_epoch = 0
    history = model.fit_generator(gtrain, 
                        steps_per_epoch=100,
                        validation_data = gvalid, 
                        validation_steps = 10,
                        initial_epoch = initial_epoch,
                        epochs = args.epochs + initial_epoch,
                        verbose = args.verbose)

    # Send metrics to sherpa.
    metrics = history.history
    sherpa.send_metrics(index=args.index, metrics=metrics, metricsfile=args.metricsfile)
    
    if 'modelfile' in args:
        # Save model file.
        model.save(args.modelfile)
        
    return

if __name__=='__main__':
    main(args)

