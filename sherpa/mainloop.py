from __future__ import absolute_import
from __future__ import division
import time
import importlib
import pickle as pkl
import os
import sys
import re
from collections import defaultdict
from .resultstable import ResultsTable
from .scheduler import SGEScheduler,LocalScheduler

class MainLoop():
    """
    Main Loop:
    1) Query Algorithm
    2) Start Experiment (possibly asynchronously)
    3) Write Results into the ResultsTable
    """

    def __init__(self, filename, algorithm, dir='./', loss='loss',
                 results_table=None, environment=None, submit_options=''):
        assert isinstance(dir, str)
        self.filename = filename  # Module file with method main(run_id, hp) (e.g. nn.py).
        self.algorithm = algorithm  # Instantiated Sherpa Algorithm object.
        self.loss = loss  # Key in Keras history to be minimized, e.g. 'loss' or 'val_loss'.
        self.dir = dir  # Directory in which all files are stored: models, history, and other output.
        # Arguments for parallel jobs.
        self.environment = environment  # Bash script that sets environment variables for parallel jobs.
        self.submit_options = submit_options  # Command line options for submission to queueing systems for parallel jobs.
        self.job_management = 'sge'  # Used for parallel jobs. Options: 'sge' or 'local'
        # Make dir if neccessary.
        try:
            os.makedirs(self.dir)  # os.makedirs(os.path.dirname(self.dir))
        except:
            print('\nWARNING: Found existing directory %s, algorithm may make '
                  'unintended use of old results!' % self.dir)
            pass

        self.results_table = results_table if results_table is not None else ResultsTable(
            self.dir, recreate=False)
        return       

    def run(self, max_concurrent=1):
        # Run main loop.
        if max_concurrent > 1:
            self.run_parallel(max_concurrent=max_concurrent)
        # Sequential loop.
        module = importlib.import_module(
            self.filename.rsplit('.', 1)[0])  # Must remove '.py' from file path.
        while True:
            # Query Algorithm
            rval = self.algorithm.next(self.results_table, pending=[])
            if rval == 'stop':
                break  # Done
            elif rval == 'wait':
                raise Exception('Should not have to wait in sequential mode.')
            else:
                assert type(rval) == tuple and len(rval) == 3
                run_id, hp, epochs = rval
                modelfile, historyfile = self.id2filenames(run_id)
                rval = module.main(modelfile=modelfile, historyfile=historyfile, hp=hp, epochs=epochs, verbose=1)
                # Update ResultsTable.
                with open(historyfile, 'rb') as f:
                    history = pkl.load(f)
                lowest_loss   = min(history[self.loss])
                epochs_seen   = len(history[self.loss])
                self.results_table.set(run_id=run_id, hp=hp, loss=lowest_loss, epochs=epochs_seen)

    def run_parallel(self, max_concurrent=1, scheduler=None):
        # Use multiprocessing to run jobs in subprocesses.
        self.id2hp     = {}  # Maps run_id to hp.
        self.scheduler = scheduler or \
                         SGEScheduler(dir=self.dir, filename=self.filename,
                                      environment=self.environment,
                                      submit_options=self.submit_options)
        while True:
            # Collect any results in the queue and write directly to ResultsTable.
            self._collect_results()

            # Check pending processes.
            pending = list(self.scheduler.active_processes.keys())
            
            # Limit number of concurrent subprocesses.
            if len(pending) >= max_concurrent:
                time.sleep(5)
                continue

            # Query Algorithm about next experiment.
            rval = self.algorithm.next(self.results_table,pending=pending)
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
                run_id, hp, epochs = rval
                modelfile, historyfile = self.id2filenames(run_id)
                self.scheduler.start_subprocess(run_id, hp, epochs, modelfile, historyfile)
                self.id2hp[run_id] = hp
                time.sleep(3)  # Delay might avoid errors in gpu locking.
                assert len(self.scheduler.active_processes) <= max_concurrent
        assert len(self.scheduler.active_processes) == 0, (self.scheduler.active_processes, list(self.scheduler.active_processes.keys()) )
        assert self.scheduler.queue_is_empty()

    def _collect_results(self):
        results = self.scheduler.read_queue() # Updates self.processes.
        for run_id in results:
            assert type(run_id) == str, run_id
            # Read historyfile to update results_table.
            modelfile, historyfile = self.id2filenames(run_id)
            with open(historyfile, 'rb') as f:
                history = pkl.load(f)
            lowest_loss = min(history[self.loss])
            epochs_seen = len(history[self.loss])
            self.results_table.set(run_id=run_id, hp=self.id2hp[run_id],
                                   loss=lowest_loss, epochs=epochs_seen)

    def id2filenames(self, run_id):
        modelfile   = os.path.join(self.dir, '{}_model.h5'.format(run_id))
        historyfile = os.path.join(self.dir, '{}_history.pkl'.format(run_id))
        return modelfile, historyfile 


# def get_hist(hp, historyfile):
#     if hp is None or len(hp) == 0:
#         # Restart from modelfile and historyfile.
#         with open(historyfile, 'rb') as f:
#             history = pkl.load(f)
#         initial_epoch = len(history['loss'])
#     else:
#         # Create new model.
#         history = defaultdict(list)
#         initial_epoch = 0
#     return history, initial_epoch
#
#
# def store_hist(partialh, history, historyfile):
#     # partialh = partialh.history
#     for k in partialh:
#         history[k].extend(partialh[k])
#     assert 'loss' in history, 'Sherpa requires a loss to be defined in history.'
#
#     with open(historyfile, 'wb') as fid:
#         pkl.dump(history, fid)
