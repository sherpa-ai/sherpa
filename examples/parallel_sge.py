# Demo for Sherpa Parallel optimization.
# Author: Peter Sadowski

# Before importing keras, decide which gpu to use. May find nothing acceptible and fail.
import sys, os
#if __name__=='__main__':
if True:
    # Don't need to lock the gpu if we are just starting Sherpa.
    os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device=cpu,floatX=float32,force_device=True,base_compiledir=~/.theano/cpu"
else:
    # Lock gpu.
    import socket
    import gpu_lock
    gpuid = gpu_lock.obtain_lock_id() # Return gpuid, or -1 if there was a problem.
    #gpuid = 0 # Force gpu0
    assert gpuid >= 0, 'No gpu available.'
    print 'Running from GPU %s' % str(gpuid)
    os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device=gpu%d,floatX=float32,force_device=True,base_compiledir=~/.theano/%s_gpu%d" % (gpuid, socket.gethostname(), gpuid)

#import h5py
import pickle as pkl
import glob
from collections import defaultdict

import keras
from keras.models import Model
from keras.layers import *
from keras.activations import *
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.utils import np_utils

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

def fit(model, hp, initial_epoch, epochs, verbose):
    '''
    ---------------------------------------------------------------------------
    USER EDITS THIS METHOD
    ---------------------------------------------------------------------------
    Update or initialize model.
    Input: 
        model   = Instantiated model or None if needs to be instantiated.
        initial_epoch = If 0, initialize the model.  
        epochs  = Number of epochs to train here.
        verbose = Pass this to fit_generator.
    Returns: 
        -(Partially) trained model.
        -(Partial) history.
    '''
    if model is None or initial_epoch == 0:
        # Initialize model.
        model = define_model(hp=hp)
    # Define dataset.
    gtrain  = dataset_bianchini(batchsize=100, nin=2, nt=3)
    # Train model
    history = model.fit_generator(gtrain, steps_per_epoch=100, epochs=epochs+initial_epoch, initial_epoch=initial_epoch, verbose=verbose)
    return model, history

def main(modelfile, historyfile, hparams={}, epochs=1, verbose=2):
    '''
    ---------------------------------------------------------------------------
    DO NOT EDIT (Edit fit method instead).
    ---------------------------------------------------------------------------
    This main function is called by Sherpa. No return value is given, 
    but it updates modelfile and historyfile.
    Input: 
        run_id     = Name of experiment.
        hparams    = Dictionary of hyperparameters.
        epochs     = Number of epochs to train this round. 
        verbose    = Passed to keras.fit_generator.
    Output:
        None
    '''
    
    if hparams is None or len(hparams) == 0:
        # Restart from modelfile and historyfile.
        model   = keras.models.load_model(modelfile)
        history = pkl.load(open(historyfile, 'r'))
        initial_epoch = len(history['loss']) 
    else:
        model  = None
        initial_epoch = 0
    
    # Update (or initialize) model. 
    model, partialh = fit(model=model, hp=hparams, initial_epoch=initial_epoch, epochs=epochs, verbose=verbose) 
    
    # Update history
    partialh = partialh.history
    if initial_epoch == 0:
        history = partialh
    else:
        for k in history:
            history[k].extend(partialh[k])
    assert 'loss' in history

    # Save files.
    model.save(modelfile)
    with open(historyfile, 'w') as fid:
        pkl.dump(history, fid)       

    return

def optimize():
    ''' 
    Run Sherpa hyperparameter optimization.
    User may want to run this as a separate file.
    '''
    import sherpa
    from sherpa.core import Hyperparameter
    import sherpa.mainloop
    import sherpa.algorithms
    import sherpa.hparam_generators

    # Specify filename that contains main method. This file contains example. 
    fname = os.path.basename(__file__) #'nn.py'
 
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
    hp_generator =  sherpa.hparam_generators.LatinHypercube
    #hp_generator =  sherpa.hparam_generators.RandomGenerator
    
    # Algorithm used for optimization.
    alg  = sherpa.algorithms.Hyperhack(samples=4, epochs_per_stage=2, stages=4, survival=0.5, hp_generator=hp_generator, hp_ranges=hp_ranges, max_concurrent=10)
    #alg  = sherpa.algorithms.Hyperband(R=3, eta=20, hpspace=hpspace)
    #alg  = sherpa.algorithms.RandomSearch(samples=100, epochs=1, hp_ranges=hp_ranges, max_concurrent=10)

    dir         = './debug' # All files written to here.
    environment = '/home/pjsadows/profiles/auto.profile' # Specify environment variables. 
    submit_options = '-N myjob -P turbomole_geopt.p -q arcus.q -l hostname=\'(arcus-1|arcus-2|arcus-3)\'' # SGE options.
    loop = sherpa.mainloop.MainLoop(fname=fname, algorithm=alg, dir=dir, environment=environment, submit_options=submit_options)
    loop.run_parallel(max_concurrent=2) # Parallel version using SGE.
    #loop.run() # Sequential.

if __name__=='__main__':
    #main() # Single run. 
    optimize() # Sherpa optimization.

