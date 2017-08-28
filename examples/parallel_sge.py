# Demo for Sherpa Parallel optimization.
# Author: Peter Sadowski
# Edits: Lars Hertel, Julian Collado
from __future__ import print_function
import sys, os
import pickle as pkl
import glob
from collections import defaultdict
import numpy as np
import gpu_lock

import sherpa
import sherpa.mainloop
import sherpa.algorithms
import sherpa.samplers
from sherpa.core import Hyperparameter



# Before importing keras, decide which gpu to use. May find nothing acceptible and fail.
#BACKEND = 'tensorflow'
BACKEND = 'theano'
if __name__=='__main__':
    # Don't use gpu if we are just starting Sherpa.
    if BACKEND == 'theano':
        os.environ['KERAS_BACKEND'] = "theano"
        os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device=cpu,floatX=float32,force_device=True,base_compiledir=~/.theano/cpu"
    elif BACKEND == 'tensorflow':
        os.environ['KERAS_BACKEND'] = "tensorflow"
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
else:
    # Lock gpu.
    import socket
    import gpu_lock
    GPUIDX = gpu_lock.obtain_lock_id() # Return gpuid, or -1 if there was a problem.
    #GPUIDX = 1
    assert GPUIDX >= 0, '\nNo gpu available.'
    print('\nRunning from GPU %s' % str(GPUIDX))
    if BACKEND == 'theano':
        os.environ['KERAS_BACKEND'] = "theano"
        os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device=gpu%d,floatX=float32,force_device=True,base_compiledir=~/.theano/%s_gpu%d" % (GPUIDX, socket.gethostname(), GPUIDX)
    elif BACKEND == 'tensorflow':
        os.environ['KERAS_BACKEND'] = "tensorflow"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUIDX)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        CONFIG = tf.ConfigProto(device_count = {'GPU': 1}, log_device_placement=False, allow_soft_placement=False) 
        CONFIG.gpu_options.allow_growth = True # Prevents tf from grabbing all gpu memory.
        sess = tf.Session(config=CONFIG)
        from keras import backend as K
        K.set_session(sess)

def dataset_bianchini(batchsize, nin=2, nt=1):
    # Dataset where we can control betti numbers.
    # Synthetic dataset where we can control Betti numbers with parameter nt.
    # Input: 2D real values, Output: binary {0,1}
    # x in [0,1]^nin, f = g(t_nt(x)) where g=1-||x||^2, t=[1-2*x_1^2,..., 1-2*x_i^2], t_nt = t(t(t(...)))
    assert nin==2
    g = lambda x: 1. - np.linalg.norm(x, ord=2)**2
    t = lambda x: 1. - 2.*(x**2)
    def f(x):
        for i in range(nt):
            x = t(x)
        return g(x)
    while True:
        X = np.random.uniform(low=-1.,high=1.0 , size=(batchsize, nin))
        Y = (np.apply_along_axis(f, axis=1, arr=X) > 0.0).astype('float32')
        yield {'input':X}, {'output':Y}

def define_model(hp):
    # Return compiled model with specified hyperparameters.
    # Model Architecture
    from keras.models import Model
    from keras.layers import Dense, Input
    from keras.optimizers import SGD
    nin    = 2
    nout   = 1
    nhidu  = hp['nhid']
    nhidl  = hp['nlayers'] - 1
    act    = hp['act']
    init   = hp['init'] 
    input  = Input(shape=(nin,), dtype='float32', name='input')
    x      = input
    for i in range(nhidl):
        x  = Dense(nhidu, kernel_initializer=init, activation=act)(x)
    output = Dense(nout, kernel_initializer=init, activation='sigmoid', name='output')(x)
    model  = Model(inputs=input, outputs=output)

    # Learning Algorithm
    lrinit    = hp['lrinit']
    momentum  = hp['momentum'] # 0.#9
    lrdecay   = hp['lrdecay'] #0.00001
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
    This main function is called by Sherpa. No return value is given,
    but it updates model_file and history_file.
    Input:
        modelfile  = File containing model.
        historyfile= File containing dictionary of per-epoch results.
        hp         = Dictionary of hyperparameters.
        epochs     = Number of epochs to train this round.
        verbose    = Passed to keras.fit_generator.
    Output:
        A list of losses or a dictionary of lists that describe history data
        to be stored
    """
    print('Running with {}'.format(str(hp)))
    import keras
    if hp is None or len(hp) == 0:
        # Restart from modelfile and historyfile.
        model = keras.models.load_model(modelfile)
        with open(historyfile, 'rb') as f:
            history = pkl.load(f)
        initial_epoch = len(history['loss'])
    else:
        # Create new model.
        model = define_model(hp=hp)
        history = defaultdict(list)
        initial_epoch = 0

    # Define dataset.
    gtrain = dataset_bianchini(batchsize=100, nin=2, nt=3)
    gvalid = dataset_bianchini(batchsize=100, nin=2, nt=3)

    # DEBUG
    if False:
        print('Training model.')
        history['loss'] = [0.1]
        history['kl']   = [0.2]
        pkl.dump(history, open(historyfile, 'wb'))
        model.save(modelfile)
        return 
    
    model.fit_generator(gtrain, steps_per_epoch=100,
                                          validation_data=gvalid, validation_steps=10,
                                          epochs=epochs + initial_epoch,
                                          initial_epoch=initial_epoch,
                                          verbose=verbose)

    # Update history
    partialh = model.history.history
    for k in partialh:
        history[k].extend(partialh[k])
    assert 'loss' in history, 'Sherpa requires a loss to be defined in history.'

    # Save model and history files.
    model.save(modelfile)
    with open(historyfile, 'wb') as fid:
        pkl.dump(history, fid)

    return

def run_example():
    filename = os.path.basename(__file__) #'nn.py'
    dir = './debug' # All files written to here.
    env = '/home/pjsadows/profiles/auto.profile' # Script specifying environment variables.
    #opt = '-N myjob -P turbomole_geopt.p -q arcus.q -l hostname=\'(arcus-1|arcus-2|arcus-3)\'' # SGE options.
    opt = '-N myjob -P turbomole_geopt.p -q arcus.q -l hostname=\'(arcus-2)\'' # SGE options.

    # Iterate algorithm accepts dictionary containing lists of possible values. 
    hp_space = {'lrinit':[0.1, 0.01],
                'act':['tanh', 'relu']}
    alg = sherpa.algorithms.Iterate(epochs=2, hp_ranges=hp_space)
    loop = sherpa.mainloop.MainLoop(filename=filename, algorithm=alg, dir=dir, environment=env, submit_options=opt)
    #loop.run() # Runs locally.
    loop.run_parallel(max_concurrent=2) # Uses SGE.
    

def run_example_advanced():
    ''' 
    Run Sherpa hyperparameter optimization.
    User may want to run this as a separate file.
    '''
    # Hyperparameter space. 
    hp_ranges = [
                 Hyperparameter(name='lrinit', distribution='choice', distr_args=[(0.1, 0.01, 0.001)]),
                 Hyperparameter(name='lrdecay', distribution='choice', distr_args=[(0.0,)]),
                 Hyperparameter(name='momentum', distribution='choice', distr_args=[(0.0, 0.5, 0.9)]),
                 Hyperparameter(name='init', distribution='choice', distr_args=[('glorot_normal', 'glorot_uniform')]),
                 Hyperparameter(name='nhid', distribution='choice', distr_args=[(20, 50, 100)]),
                 Hyperparameter(name='nlayers', distribution='choice', distr_args=[(2, 3, 4)]),
                 Hyperparameter(name='act', distribution='choice', distr_args=[('tanh','relu')]),
                ]
    
    # Specify how initial hp combinations are sampled.
    sampler =  sherpa.samplers.LatinHypercube
    #sampler =  sherpa.samplers.RandomGenerator
    
    # Algorithm used for optimization.
    alg  = sherpa.algorithms.Hyperhack(samples=4, epochs_per_stage=2, stages=4, survival=0.5, sampler=sampler, hp_ranges=hp_ranges, max_concurrent=10)
    #alg  = sherpa.algorithms.Hyperhack(samples=4, epochs_per_stage=2, stages=4, survival=0.5, hp_ranges=hp_ranges, max_concurrent=10)
    #alg  = sherpa.algorithms.Hyperband(R=3, eta=20, hpspace=hpspace)
    #alg  = sherpa.algorithms.RandomSearch(samples=100, epochs=1, hp_ranges=hp_ranges, max_concurrent=10)

    # Specify filename that contains main method. This file contains example. 
    filename = os.path.basename(__file__) #'nn.py'
    
    # Parallel options. 
    dir         = './debug' # All files written to here.
    environment = '/home/pjsadows/profiles/auto.profile' # Specify environment variables.
    submit_options = '-N myjob -P turbomole_geopt.p -q arcus.q -l hostname=\'(arcus-1|arcus-2|arcus-3)\'' # SGE options.
    
    loop = sherpa.mainloop.MainLoop(filename=filename, algorithm=alg, dir=dir, environment=environment, submit_options=submit_options)
    #from sherpa.scheduler import LocalScheduler
    #loop.run_parallel(max_concurrent=2, scheduler=LocalScheduler(filename=filename)) # Parallel version using SGE.
    loop.run_parallel(max_concurrent=1)
    #loop.run() # Sequential.

if __name__=='__main__':
    #main() # Single run.
    run_example() # Sherpa optimization.
    #run_example_advanced() # Sherpa optimization.

