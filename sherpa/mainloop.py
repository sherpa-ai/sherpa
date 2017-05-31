from __future__ import absolute_import
from __future__ import division
#from .experiment import Experiment
from .resultstable import ResultsTable
from .core import Hyperparameter
import keras.backend as K
from time import time, sleep
import importlib
import pickle as pkl
import os

def get_prefix(run_id, hparams=None, history=None):
    # Simply return file prefix fo
    return 

class MainLoop():
    """
    Main Loop:
    1) Query Algorithm
    2) Start Experiment (possibly asynchronously)
    3) Write Results into the ResultsTable
    """
    def __init__(self, fname, algorithm, dir='./', loss='loss', results_table=None):
        assert isinstance(dir, str)
        self.fname     = fname     # e.g. nn.py with main(run_id, hparams)
        self.algorithm = algorithm
        self.loss      = loss      # Keras history channel e.g. 'loss' or 'val_loss' to be minimized.
        self.dir       = dir
        self.pending = {}  # Keep track of experiments that aren't finished.
        
        # Make dir if neccessary.
        try:
            os.makedirs(self.dir)
            #os.makedirs(os.path.dirname(self.dir))
        except:
            print '\nWARNING: Found existing directory %s, algorithm may not be able to make sense of old results!' % self.dir
            pass
        
        self.results_table = results_table if results_table is not None else ResultsTable(dir, recreate=False)
        
        return

    def run(self, max_concurrent=1):
         
        assert max_concurrent == 1, max_concurrent
        if max_concurrent > 1:
            self.run_parallel(max_concurrent=max_concurrent)
        
        while True:
            # Query Algorithm 
            rval = self.algorithm.next(self.results_table, self.pending)
            if rval == 'stop':
                break # Done
            elif rval == 'wait':
                sleep(5) # Wait for other jobs to finish.
                continue
            else:
                assert type(rval) == tuple and len(rval) == 3
            run_id, hparams, epochs = rval
            
            # Run Experiment
            modelfile   = '%s/%s_model.h5' % (self.dir, run_id)
            historyfile = '%s/%s_history.pkl' % (self.dir, run_id)
            checkpoint  = (modelfile, historyfile) if os.path.isfile(modelfile) else None
            module      = importlib.import_module(self.fname.rsplit('.', 1)[0]) # Must remove '.py' from file path.
            model, history = module.main(run_id=run_id, hparams=hparams, epochs=epochs, checkpoint=checkpoint, verbose=1)
            # Save results.
            model.save(modelfile)
            with open(historyfile, 'w') as fid:
                pkl.dump(history, fid)            
            
            #  Update results table.
            lowest_loss   = min(history[self.loss])
            epochs_seen   = len(history[self.loss])
            self.results_table.set(run_id=run_id, hparams=hparams, loss=lowest_loss, epochs=epochs_seen)
            
        print 'Done'
       
