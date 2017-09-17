# Demo for Sherpa Parallel optimization.
# Author: Peter Sadowski
# Edits: Lars Hertel, Julian Collado
from __future__ import print_function
import sys, os
import socket
import numpy as np
import glob
import pickle as pkl
from collections import defaultdict

import sherpa
from sherpa.resultstable import ResultsTable
from sherpa.hyperparameters import Hyperparameter
from sherpa.scheduler import LocalScheduler,SGEScheduler

os.environ['KERAS_BACKEND'] = 'theano' # Or 'tensorflow'
if __name__=='__main__':
    # Don't use gpu if we are just starting Sherpa. 
    if os.environ['KERAS_BACKEND'] == 'theano':
        os.environ['THEANO_FLAGS'] = "floatX=float32,device=cpu,base_compiledir=~/.theano/cpu"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
else:
    # Before importing keras, decide which gpu to use.
    try:
        # gpu_lock module located at /home/pjsadows/libs
        import gpu_lock
        GPUIDX = gpu_lock.obtain_lock_id() # Return gpuid, or -1 if there was a problem.
    except:
        print('Could not import gpu_lock. Prepend /extra/pjsadows0/libs/shared/gpu_lock/ to PYTHONPATH.')
        GPUIDX = 0
    assert GPUIDX >= 0, '\nNo gpu available.'
    print('\nRunning from GPU %s' % str(GPUIDX))
    # Carefully import backend.
    if os.environ['KERAS_BACKEND'] == 'theano':
        #os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device=gpu{},floatX=float32,force_device=True,base_compiledir=~/.theano/{}_gpu{}".format(GPUIDX, socket.gethostname(), GPUIDX)
        os.environ['THEANO_FLAGS'] = "floatX=float32,device=cuda{},base_compiledir=~/.theano/{}_gpu{}".format(GPUIDX, socket.gethostname(), GPUIDX)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUIDX)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        CONFIG = tf.ConfigProto(device_count = {'GPU': 1}, log_device_placement=False, allow_soft_placement=False) 
        CONFIG.gpu_options.allow_growth = True # Prevents tf from grabbing all gpu memory.
        sess = tf.Session(config=CONFIG)
        from keras import backend as K
        K.set_session(sess)
    import keras

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

def define_model(hp):
    '''
    Return compiled model using hyperparameters specified in dict hp.
    ''' 
    from keras.models import Model
    from keras.layers import Dense, Input
    from keras.optimizers import SGD
    nin    = 2
    nout   = 1
    units  = 10
    nhlay  = 2
    act    = hp['act']
    init   = 'glorot_normal'
    input  = Input(shape=(nin,), dtype='float32', name='input')
    x      = input
    for units in hp['arch']:
        x  = Dense(units, kernel_initializer=init, activation=act)(x)
    output = Dense(nout, kernel_initializer=init, activation='sigmoid', name='output')(x)
    model  = Model(inputs=input, outputs=output)

    # Learning Algorithm
    lrinit    = hp['lrinit']
    momentum  = hp['momentum']
    lrdecay   = hp['lrdecay'] 
    loss      = {'output':'binary_crossentropy'}
    metrics   = {'output':'accuracy'}
    loss_weights = {'output':1.0}
    optimizer = SGD(lr=lrinit, momentum=momentum, decay=lrdecay)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights)
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
        initial_epoch = len(history['loss']) # Assumes loss is list of length epochs.
    else:
        # Create new model.
        model   = define_model(hp=hp)
        history = defaultdict(list)
        initial_epoch = 0

    print('Running with {}'.format(str(hp)))
    
    # Define dataset.
    gtrain = dataset_bianchini(batchsize=100, k=3)
    gvalid = dataset_bianchini(batchsize=100, k=3)

    model.fit_generator(gtrain, 
                        steps_per_epoch=100,
                        validation_data = gvalid, 
                        validation_steps = 10,
                        epochs = epochs + initial_epoch,
                        initial_epoch = initial_epoch,
                        verbose = verbose)

    # Update history and save to file.
    partialh = model.history.history
    for k in partialh:
        history[k].extend(partialh[k])
    with open(historyfile, 'wb') as fid:
        pkl.dump(history, fid)
    # Save model file if we want to restart.
    model.save(modelfile)

    return

def run_example():
    '''
    Run parallel Sherpa optimization over a set of discrete hp combinations.
    '''
    # Iterate algorithm accepts dictionary containing lists of possible values. 
    hp_space = {'act':['tanh'],#, 'relu'],
                'lrinit':[0.1],#[0.1, 0.01],
                'momentum':[0.0],
                'lrdecay':[0.0],
                'arch': [[20]],
                }
    alg = sherpa.algorithms.Iterate(epochs=2, hp_ranges=hp_space)
    f   = os.path.basename(__file__) # The 'main' function of this file is called.
    dir = './output' # All files written to here.
    env = '/home/pjsadows/profiles/auto.profile' # Script specifying environment variables.
    opt = '-N myexample -P arcus.p -q arcus-ubuntu.q -q arcus.q -l hostname=\'(arcus-1|arcus-2)\'' # SGE options.
    #sched = None # Serial mode.
    sched = LocalScheduler() # Run on local machine without SGE.
    #sched = SGEScheduler(dir=dir, environment=env, submit_options=opt)
    rval = sherpa.optimize(filename=f, algorithm=alg, dir=dir, overwrite=True, scheduler=sched, max_concurrent=2)
    print()
    print('Best results:')
    print(rval)

def run_example_advanced():
    ''' 
    Run Sherpa hyperparameter optimization.
    User may want to run this as a separate file.
    '''
    # Hyperparameter space. 
    hp_space = [
                 Hyperparameter(name='lrinit', distribution='choice', dist_args=[(0.1, 0.01, 0.001)]),
                 Hyperparameter(name='lrdecay', distribution='choice', dist_args=[(0.0,)]),
                 Hyperparameter(name='momentum', distribution='choice', dist_args=[(0.0, 0.5, 0.9)]),
                 Hyperparameter(name='act', distribution='choice', dist_args=[('tanh','relu')]),
                ]
    
    # Specify how initial hp combinations are sampled.
    sampler =  sherpa.samplers.LatinHypercube # Or sherpa.samplers.RandomGenerator
    
    # Algorithm used for optimization.
    alg  = sherpa.algorithms.Hyperhack(samples=4, epochs_per_stage=2, stages=4, survival=0.5, sampler=sampler, hp_ranges=hp_space)
    #alg  = sherpa.algorithms.RandomSearch(samples=100, epochs=1, hp_ranges=hp_ranges, max_concurrent=10)
    
    f   = os.path.basename(__file__) # The 'main' function of this file is called.
    dir = './output' # All files written to here.
    env = '/home/pjsadows/profiles/auto.profile' # Script specifying environment variables.
    opt = '-N myexample -P arcus.p -q arcus-ubuntu.q -q arcus.q -l hostname=\'(arcus-1|arcus-2)\'' # SGE options.
    sched = SGEScheduler(dir=dir, environment=env, submit_options=opt)
    rval = sherpa.optimize(filename=f, algorithm=alg, dir=dir, overwrite=True, scheduler=sched, max_concurrent=4)
    print()
    print('Best results:')
    print(rval)

if __name__=='__main__':
    run_example() # Sherpa optimization.
    #run_example_advanced() # Sherpa optimization.

