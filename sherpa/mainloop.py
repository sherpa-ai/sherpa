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
    scheduler = Sherpa Scheduler object, defaults to LocalScheduler with single process (serial mode).
    max_concurrent = Limits the number of jobs Sherpa submits to scheduler. 
    '''
    loop = MainLoop(filename, algorithm, dir=dir, results_table=results_table, loss=loss, overwrite=overwrite) 
    loop.run(scheduler=scheduler, max_concurrent=max_concurrent) 
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
    metricsfile, and letting the MainLoop know that the calculation is done.
    The MainLoop then tells the ResultsFile to update itself with the results.

    Details:
    ResultsTable: Should only be modified at initialization and by 'update' 
                  method called from the MainLoop. 

    Algorithm: Should not modify ResultsTable. 

    TODO: Figure out a way to clearly make use of existing results if desired. 

    """

    def __init__(self, filename, algorithm, dir='./output/', results_table=None, overwrite=False, loss='loss', loss_summary=None):
        assert isinstance(dir, str)
        self.filename = filename  # Module file with method main(index, hp) (e.g. nn.py).
        self.algorithm = algorithm  # Instantiated Sherpa Algorithm object.
        self.dir = dir 
        self.dir_models  = os.path.join(self.dir, 'models') # Directory in which model files are (optionally) stored.
        self.dir_metrics = os.path.join(self.dir, 'metrics') # Directory in which metrics files are stored.
        
        # Clear out existing directories.
        for d in [self.dir_metrics, self.dir_models]:
            if not os.path.isdir(d):
                os.makedirs(d)
            else:
                if not overwrite:
                    print('WARNING: Found existing directory {}, algorithm may make '
                          'unintended use of old results!'.format(d))
                else:
                    print('WARNING: Overwriting all files in {}!'.format(d))
                    for f in glob.glob(os.path.join(d, '*')):
                        os.remove(f)
       
       
        self.results_table = results_table or ResultsTable(self.dir, loss=loss, loss_summary=loss_summary)
  
        return       

    def run(self, scheduler=None, max_concurrent=1):
        # Use multiprocessing to run jobs in subprocesses.
        self.scheduler = scheduler or LocalScheduler()
        self.scheduler.set_dir(self.dir)
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
                if type(rval) != dict:
                    raise ValueError('Algorithm.next() should return "stop", "wait", or dict of hyperparams. Returned {}'.format(rval))
                # Start new experiment specified by Algorithm.
                hp = rval
                index = self.results_table.on_start(hp=hp) # ResultsTable returns unique index.
                modelfile, metricsfile = self.id2filenames(index) # TODO: Maybe we want to restart models and save to same file. 
                # Submit experiment to scheduler.
                self.scheduler.start_subprocess(self.filename, index, hp, modelfile, metricsfile) 
                time.sleep(3)  # Delay might avoid errors in gpu locking.
                assert len(self.scheduler.get_active_processes()) <= max_concurrent
        assert self.scheduler.queue_is_empty()
        assert len(self.scheduler.get_active_processes()) == 0

    def _collect_results(self):
        results = self.scheduler.get_all_from_queue() # Updates self.processes.
        for index in results:
            # Read metricsfile to update results_table.
            modelfile, metricsfile = self.id2filenames(index)
            self.results_table.on_finish(index=index, historyfile=metricsfile)

    def id2filenames(self, index):
        modelfile   = os.path.join(self.dir_models, '{}.h5'.format(index))
        metricsfile = os.path.join(self.dir_metrics, '{}.pkl'.format(index))
        return modelfile, metricsfile 


