# Demo for Sherpa bayesian optimization using the antihydrogen calibration dataset 


from __future__ import print_function
import sys, os
import socket
import numpy as np
import glob
import pickle as pkl
import _pickle as cpkl

from collections import defaultdict

import sherpa
from sherpa.resultstable import ResultsTable
from sherpa.hyperparameters import DistributionHyperparameter as Hyperparameter
from sherpa.scheduler import LocalScheduler,SGEScheduler
import h5py

os.environ['KERAS_BACKEND'] =  'tensorflow' #'theano' # Or
if __name__=='__main__':
    # Don't use gpu if we are just starting Sherpa. 
    if os.environ['KERAS_BACKEND'] == 'theano':
        os.environ['THEANO_FLAGS'] = "floatX=float32,device=cpu,base_compiledir=~/.theano/cpu"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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
        os.environ['THEANO_FLAGS'] = "floatX=float32,device=cuda{},base_compiledir=~/.theano/{}_gpu{}".format(GPUIDX, socket.gethostname(), GPUIDX)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUIDX)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        CONFIG = tf.ConfigProto(device_count = {'GPU': 1}, log_device_placement=False, allow_soft_placement=False) 
        #CONFIG.gpu_options.allow_growth = True # Prevents tf from grabbing all gpu memory.
        CONFIG.gpu_options.per_process_gpu_memory_fraction = 0.4

        sess = tf.Session(config=CONFIG)
        from keras import backend as K
        K.set_session(sess)
    import keras

#### data generator
from scipy import ndimage
import time
def shift(xs):
    e = np.zeros( (xs.shape[0],446*2+1) ) #893
    for i in range(e.shape[0]):
        if sum(xs[i,:]) == 0:
            e[i,:] = 0
        else:
            n = int(ndimage.measurements.center_of_mass(xs[i,:])[0]) # center of mass
            e[:,(446-n):(893-n)] = xs
        return e
    
from scipy.ndimage.interpolation import zoom



def data_generator(filename, batchsize, enforce_batchsize=False, start=0, translate = False):
    with h5py.File(filename, 'r') as f:
        stopa =  f['features_a'].shape[0] #761400#
        stopb = f['features_b'].shape[0] #488900 #
        iexamplea, iexampleb = 0 , 0
        batchsize_half = 50 #int(batchsize/2)
        while True:
            batcha = slice(iexamplea,iexamplea + batchsize_half)# min(iexamplea + batchsize_half, stopa))
            X_a  = f['features_a'][batcha,7:]
            batchb = slice(iexampleb, iexampleb + batchsize_half)#min(iexampleb + batchsize_half, stopb))
            X_b  = f['features_b'][batchb,7:] 

            ### divide into two parts: azimuthal (246+ 290) and axial(447*2)
            X = np.vstack([X_a,X_b])
            Y = np.hstack([np.zeros(batchsize_half), np.ones(batchsize_half)]).reshape(100,1)

            iz   = X[:, 0:447]
            iphi = X[:, 447:447+246]
            oz   = X[:, 447+246:-290]
            ophi = X[:, -290:]
            ophi = zoom(ophi, zoom=(1., 246./290))
            if translate == True:
                iz = shift(iz) 
                oz = shift(oz)
                X1 = np.asarray([iphi, ophi])
                X1 = np.transpose( X1, (1,2,0))
                X2 = np.asarray([iz, oz])
                X2 = np.transpose( X2, (1,2,0) )          
                yield [X1,X2], Y

                iexamplea += batchsize_half
                iexampleb += batchsize_half
                if iexamplea +50 >= stopa:
                    iexamplea = 0 #start   
                if iexampleb +50 >= stopb:
                        iexampleb = 0 #start 


def define_model(hp):
    '''
    Return compiled model using hyperparameters specified in dict hp.
    ''' 

    from keras.models import Sequential, Model # Use Keras functional API
    from keras.layers import Input, Dense, Flatten, Reshape, merge, Activation, \
            Convolution2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D, Dropout, MaxPooling1D
    from keras.layers import merge  ### To merge layers
    from keras.layers import Merge  ### To merge layers

    from keras import regularizers
    from keras.optimizers import SGD, Adam # , RMSprop, Adagrad,Adadelta, Adamax
    from keras.utils.io_utils import HDF5Matrix
    from keras.callbacks import Callback, EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
    from keras.layers.convolutional import Conv1D        ### add for 1D cnn __adam
    from keras.layers.convolutional import MaxPooling1D 
    
    act        = 'relu' # 'tanh' # 'relu'
    init       = 'he_normal' #'he_normal' 
    output_act = 'sigmoid'

    branch1 = Sequential()
    branch1.add(Conv1D(filters=8,kernel_size = 7, init = init, activation = act,strides = 1,
                       padding = "valid", input_shape=(246,2)) )
    branch1.add(Conv1D(filters=16,kernel_size = 3, init = init, activation = act,strides = 1,padding = "valid"))
    branch1.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
    
    branch1.add(Conv1D(filters=32,kernel_size = 3, init = init, activation = act,strides = 1,padding = "valid"))
    branch1.add(Conv1D(filters=64,kernel_size = 3, init = init, activation = act,strides = 1,padding = "valid"))
    
    branch1.add(Conv1D(filters=128,kernel_size = 3, init = init, activation = act,strides = 1,padding = "valid"))
    branch1.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
    branch1.add(Flatten())
    
########## Branch 2
   
    branch2 = Sequential()
    branch2.add(Conv1D(filters=8,kernel_size = 7, strides = 1 ,init = init, activation = act
                       ,padding = "valid",input_shape=(893,2)))
    branch2.add(Conv1D(filters=16,kernel_size = 3, init = init, activation = act,strides = 1,padding = "valid"))
    branch2.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
    
    branch2.add(Conv1D(filters=32,kernel_size = 3, init = init, activation = act,strides = 1,padding = "valid"))
    branch2.add(Conv1D(filters=64,kernel_size = 3, init = init, activation = act,strides = 1,padding = "valid"))
    branch2.add(Conv1D(filters=128,kernel_size = 3, init = init, activation = act,strides = 1,padding = "valid"))
    branch2.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
    branch2.add(Flatten())

    model = Sequential()
    model.add(Merge([branch1, branch2], mode = 'concat'))

    model.add(Dense(1000, init = init, activation = act  ))
    model.add(Dropout(0.5))
    model.add(Dense(1000, init = init, activation = act  ))
    model.add(Dropout(0.5))
    model.add(Dense(1000, init = init, activation = act  ))
    model.add(Dropout(0.5))
    model.add(Dense(1000, init = init, activation = act ))
    model.add(Dropout(0.5))
    
    model.add(Dense(1, init = init, activation = 'sigmoid'))
    
    
    lrinit   = hp['lr']
    momentum = 0.9 
    optimizer = Adam(lr=lrinit, beta_1=momentum, beta_2=0.999, epsilon=1e-08) 
    model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])
    
    
    
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
    data         = defaultdict(dict)
    data['path'] = '/home/yadongl1/17Summer/Dataset_new/exp3.0_pandas/'
    batchsize    = 100
    
    subsets      = ['train', 'valid', 'test']
    for subset in subsets:
        data[subset]['filename'] = data['path'] + 'risingedge_rmnohits_%s.h5' % subset
        with h5py.File(data[subset]['filename'], 'r') as f: # Count number of examples
            data[subset]['steps_perep'] = (f['features_a'].shape[0] + f['features_b'].shape[0])//batchsize
        data[subset]['generator'] = data_generator(data[subset]['filename'], batchsize=batchsize, translate = True)
    
    if os.path.isfile(historyfile):
        # Resume training.
        assert os.path.isfile(modelfile)
        assert hp is None or len(hp) == 0
        model = keras.models.load_model(modelfile)
        with open(historyfile, 'rb') as f:
            history_loaded = cpkl.load(f) # origin pkl.load(f)
        initial_epoch = len(history['loss']) # Assumes loss is list of length epochs.
    else:
        # Create new model.
        model   = define_model(hp=hp)
        history_loaded = defaultdict(list)
        initial_epoch = 0

    print('Running with {}'.format(str(hp)))


    history = model.fit_generator(data['train']['generator'], 
                        steps_per_epoch= data['train']['steps_perep'],
                        validation_data = data['valid']['generator'], 
                        validation_steps = data['valid']['steps_perep'],
                        epochs = epochs + initial_epoch,
                        initial_epoch = initial_epoch,
                        verbose = 1)

    # Update history and save to file.
    partialh = history.history #model.history.history
    for k in partialh:
        history_loaded[k].extend(partialh[k])
    print(history.history)
    with open(historyfile, 'wb') as fid:
        cpkl.dump(history_loaded, fid)  #pkl.dump(history, fid) 
    # Save model file if we want to restart.
    model.save(modelfile)

    return

def run_example():
    '''
    Run parallel Sherpa optimization over a set of discrete hp combinations.
    '''
    # Iterate algorithm accepts dictionary containing lists of possible values. 
    hp_space = {#'act':['tanh'],#, 'relu'],
                'lrinit':[0, 0.0001],#[0.1, 0.01],
                #'momentum':[0.0],
                #'lrdecay':[0.0],
                # 'arch': [[20]],
                }
    alg = sherpa.algorithms.GaussianEI(num_eval = 11, hp_ranges = hp_space['lrinit'])
    f   = 'Basian_demo.py' #os.path.basename(__file__) # The 'main' function of this file is called.
    dir = './output' # All files written to here.

    sched = LocalScheduler() # Run on local machine without SGE.
    #sched = SGEScheduler(dir=dir, environment=env, submit_options=opt)
    rval = sherpa.optimize(filename=f, algorithm=alg, dir=dir, overwrite=True, scheduler=sched, max_concurrent=2)
    print()
    print('Best results:')
    print(rval)

if __name__=='__main__':
    run_example() # Sherpa optimization.
    #run_example_advanced() # Sherpa optimization.

