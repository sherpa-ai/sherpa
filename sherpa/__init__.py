from __future__ import absolute_import
from . import hyperparameters
from . import samplers
from . import algorithms
from . import resultstable
from . import scheduler
from . import mainloop
from .mainloop import optimize

import pickle as pkl
def send_metrics(index, metrics, metricsfile=None, db=None):
    '''
    Called by experiment to report results.
    INPUTS:
        index = Unique experiment index (int).
        metrics = Dictionary of metricname, list pairs, 
                  where list length is assumed to correspond
                  to the number of epochs. 
    '''
    if metricsfile:
        # Save metrics to metricsfile.
        with open(metricsfile, 'wb') as fid:
            pkl.dump(metrics, fid)
    if db:
        # Send metrics to db.
        raise NotImplementedError('Database not implemented yet.') 
    if not metricsfile and not db:
        raise NotImplementedError('Method send_metrics should specify metricsfile and/or db.') 
        # Maybe create name from index?
    return
 
