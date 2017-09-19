from __future__ import absolute_import
from __future__ import division
import time
import importlib
import pickle as pkl
import os
import sys
import re
import glob
from collections import defaultdict
from .resultstable import ResultsTable
from .scheduler import SGEScheduler,LocalScheduler

def optimize(filename, algorithm, 
             dir='./output',
             results_table=None, 
             loss='loss',
             overwrite=False,
             scheduler=None, 
             max_concurrent=1): 
    ''' 
    Convenience function for running Sherpa optimization.
    INPUTS:
    filename = File containing main function that runs experiment.
    algorithm = Sherpa algorithm.
    dir      = Sherpa models are saved in (dir)/sherpa_models/.
    results_table = Sherpa ResultsTable object to use. 
    loss     = String key specifying which channel to minimize.
    overwrite = If True, deletes existing files in (dir).
    scheduler = Sherpa Scheduler object, defaults to serial mode.
    max_concurrent = Limits the number of jobs Sherpa submits to scheduler. 
    '''
    
    loop = MainLoop(filename, algorithm, dir=dir, results_table=results_table, loss=loss, overwrite=overwrite) 
    if scheduler is None:
        assert max_concurrent == 1, 'Define a scheduler for parallelization.'
        loop.run_serial() 
    else: 
        loop.run_parallel(scheduler=scheduler, max_concurrent=max_concurrent) 
    # Return best result. 
    rval = loop.results_table.get_best()
    return rval 

class MainLoop():
    """
    Main Loop:
    1) Query Algorithm
    2) Start experiment (possibly asynchronously)
    3) Write results to the 
    
    Organization Summary:
    The MainLoop is responsible for coordination between the Algorithm,
    the Scheduler, and the ResultsTable. A reference to the ResultsTable is
    given to the Algorithm so that it can recommend a set of hyperparameters,
    which the MainLoop passes to the Scheduler. The Scheduler is responsible
    for training with a set of hp, writing the results to the modelfile and
    historyfile, and letting the MainLoop know that the calculation is done.
    The MainLoop then tells the ResultsFile to update itself with the results.

    Details:
    ResultsTable: Should only be modified at initialization and by 'update' 
                  method called from the MainLoop. 

    Algorithm: Should not modify ResultsTable. 



    """

    def __init__(self, filename, algorithm, dir='./output/', results_table=None, overwrite=False, loss='loss', loss_summary=None):
        assert isinstance(dir, str)
        self.filename = filename  # Module file with method main(index, hp) (e.g. nn.py).
        self.algorithm = algorithm  # Instantiated Sherpa Algorithm object.
        self.dir = dir 
        self.dir_models = os.path.join(self.dir, 'sherpa_models') # Directory in which all model, history files are stored.
        if not os.path.isdir(self.dir_models):
            os.makedirs(self.dir_models)
        else:
            if not overwrite:
                print('\nWARNING: Found existing directory {}, algorithm may make '
                      'unintended use of old results!'.format(self.dir))
            else:
                print('\nWARNING: Overwriting all files in {} of the form '
                      '*_model.h5 and *_history.pkl!'.format(self.dir_models))
                for f in glob.glob(os.path.join(self.dir_models, '*_model.h5')):
                    os.remove(f)
                for f in glob.glob(os.path.join(self.dir_models, '*_history.pkl')):
                    os.remove(f)
        
        self.results_table = results_table or ResultsTable(self.dir, loss=loss, loss_summary=loss_summary)
  
        return       

    def run_serial(self):
        # Sequential loop.
        module = importlib.import_module(self.filename.rsplit('.', 1)[0]) # Remove .py.
        while True:
            # Query Algorithm
            rval = self.algorithm.next(self.results_table)
            if rval == 'stop':
                break  # Done
            elif rval == 'wait':
                raise Exception('Should not have to wait in sequential mode.')
            else:
                if type(rval) is not tuple:
                    raise ValueError('Algorithm.next() should return "stop", "wait", or tuple. Returned {}'.format(rval))
                if type(rval[0]) == int:
                    # Resume training of this model.
                    index, epochs = rval
                    self.results_table.on_start(index=index)
                elif type(rval[0]) == dict:
                    hp, epochs = rval
                    index = self.results_table.on_start(hp=hp) # ResultsTable returns unique index.
                    assert index not in self.id2hp
                    self.id2hp[index] = hp
                else:
                    raise ValueError('Algorithm.next()[0] should be int or dict. Returned {}'.format(rval))
                modelfile, historyfile = self.id2filenames(index)
                rval = module.main(modelfile=modelfile, historyfile=historyfile, hp=hp, epochs=epochs, verbose=1)
                # Update ResultsTable.
                self.results_table.on_finish(index=index, historyfile=historyfile)

    def run_parallel(self, scheduler=None, max_concurrent=1):
        # Use multiprocessing to run jobs in subprocesses.
        self.id2hp     = {}  # Maps index to hp.
        self.scheduler = scheduler or LocalScheduler()
        while True:
            # Collect any results in the queue and write directly to ResultsTable.
            self._collect_results()

            # Limit number of concurrent subprocesses.
            pending = self.scheduler.get_active_processes() # This should match results_table.get_pending()
            if len(pending) >= max_concurrent:
                time.sleep(5)
                continue

            # Query Algorithm about next experiment.
            rval = self.algorithm.next(self.results_table)
            if rval == 'stop' and len(pending) == 0: 
                # Finished.
                break
            elif rval == 'stop' and len(pending)>0:
                # Wait for all jobs to complete before submitting more.
                time.sleep(5)
                continue
            elif rval == 'wait' and len(pending)>0:
                # Wait for all jobs to complete before submitting more.
                time.sleep(5)
                continue
            elif rval == 'wait' and len(pending)==0:
                raise Exception('Algorithm shouldnt wait if there are no pending jobs.')
            else:
                # Start new experiment specified by Algorithm.
                if type(rval) is not tuple:
                    raise ValueError('Algorithm.next() should return "stop", "wait", or tuple. Returned {}'.format(rval))
                if type(rval[0]) == int:
                    # Resume training of this model.
                    index, epochs = rval
                    self.results_table.on_start(index=index)
                elif type(rval[0]) == dict:
                    hp, epochs = rval
                    index = self.results_table.on_start(hp=hp) # ResultsTable returns unique index.
                    assert index not in self.id2hp
                    self.id2hp[index] = hp
                else:
                    raise ValueError('Algorithm.next()[0] should be int or dict. Returned {}'.format(rval))
                #index, hp, epochs = rval
                modelfile, historyfile = self.id2filenames(index)
                self.scheduler.start_subprocess(self.filename, index, hp, epochs, modelfile, historyfile)
                time.sleep(3)  # Delay might avoid errors in gpu locking.
                assert len(self.scheduler.get_active_processes()) <= max_concurrent
        assert self.scheduler.queue_is_empty()
        assert len(self.scheduler.get_active_processes()) == 0

    def _collect_results(self):
        results = self.scheduler.get_all_from_queue() # Updates self.processes.
        for index in results:
            # Read historyfile to update results_table.
            modelfile, historyfile = self.id2filenames(index)
            self.results_table.on_finish(index=index, historyfile=historyfile)

    def id2filenames(self, index):
        modelfile   = os.path.join(self.dir_models, '{}_model.h5'.format(index))
        historyfile = os.path.join(self.dir_models, '{}_history.pkl'.format(index))
        return modelfile, historyfile 


