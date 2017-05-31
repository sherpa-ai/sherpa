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

from multiprocessing import Process, Queue

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
        self.module   = importlib.import_module(self.fname.rsplit('.', 1)[0]) # Must remove '.py' from file path.
        
        # Make dir if neccessary.
        try:
            os.makedirs(self.dir)
            #os.makedirs(os.path.dirname(self.dir))
        except:
            print '\nWARNING: Found existing directory %s, algorithm may make Byzantine use of old results!' % self.dir
            pass
        
        self.results_table = results_table if results_table is not None else ResultsTable(dir, recreate=False)
        
        return

    def run(self, max_concurrent=1):
         
        #assert max_concurrent == 1, max_concurrent
        if max_concurrent > 1:
            self.run_parallel(max_concurrent=max_concurrent)
        
        while True:
            # Query Algorithm 
            rval = self.algorithm.next(self.results_table, pending={})
            if rval == 'stop':
                break # Done
            elif rval == 'wait':
                #sleep(5) # Wait for other jobs to finish.
                #continue
                raise Exception('Shouldnt have to wait in sequential mode.')
            else:
                assert type(rval) == tuple and len(rval) == 3
                run_id, hparams, epochs = rval
                modelfile   = '%s/%s_model.h5' % (self.dir, run_id)
                historyfile = '%s/%s_history.pkl' % (self.dir, run_id)
            
            # Run Experiment
            
            rval = self.module.main(modelfile=modelfile, historyfile=historyfile, hparams=hparams, epochs=epochs, verbose=1)
                      
            # Update ResultsTable.
            history       = pkl.load(open(historyfile, 'r'))
            lowest_loss   = min(history[self.loss])
            epochs_seen   = len(history[self.loss])
            self.results_table.set(run_id=run_id, hparams=hparams, loss=lowest_loss, epochs=epochs_seen)
            
        print 'Done'
        
    def run_parallel(self, max_concurrent=1):
        # Use multiprocessing to run jobs in subprocesses.
        queue     = Queue()
        processes = {}
        while True:
            # Collect any results in the queue.
            if not queue.empty():
                run_id, rval, historyfile = queue.get()
                processes[run_id].join() # Should be done.
                processes.pop(run_id)
                # Update results table.
                history       = pkl.load(open(historyfile, 'r'))
                lowest_loss   = min(history[self.loss])
                epochs_seen   = len(history[self.loss])
                #print run_id, rval, lowest_loss
                self.results_table.set(run_id=run_id, hparams=hparams, loss=lowest_loss, epochs=epochs_seen)
            
            # Wait if we are maxed out.
            if len(processes) >= max_concurrent:
                sleep(5)
                continue
            
            # Query Algorithm 
            rval = self.algorithm.next(self.results_table, pending=processes.keys())
            if rval == 'stop':
                break # Done
            elif rval == 'wait':
                sleep(5) # Wait for jobs to finish.
                continue
            else:
                assert type(rval) == tuple and len(rval) == 3
                run_id, hparams, epochs = rval
                modelfile   = '%s/%s_model.h5' % (self.dir, run_id)
                historyfile = '%s/%s_history.pkl' % (self.dir, run_id)
            
            # Start Experiment
            def f(queue, modelfile, historyfile, hparams, epochs):
                rval = self.module.main(modelfile=modelfile, historyfile=historyfile, hparams=hparams, epochs=epochs, verbose=2)  
                queue.put([run_id, rval, historyfile]) # Should only need run_id.
            p = Process(target=f, args=(queue, modelfile, historyfile, hparams, epochs))
            p.start()
            processes[run_id] = p
            assert len(processes) <= max_concurrent
            sleep(1) # Delay might avoid errors in gpu locking.
        print 'Done'
       
       
