from __future__ import print_function
from __future__ import absolute_import
import pytest
import numpy as np
import pickle as pkl
#from sherpa.hyperparameters import DistributionHyperparameter 
#from sherpa.samplers import RandomSampler, GridSearch
from sherpa.resultstable import ResultsTable
from sherpa.algorithms import LocalSearch

@pytest.mark.run(order=1)
def test_LocalSearch():
    """
    """
    r = ResultsTable(dir='./temp', loss='loss', loss_summary=None, load_results=None)

    # Try on empty results table.
    hp_space = {'hp1':[0,1,2,3],
                'hp2':['tanh']}
    hp_init = {'hp1':0, 'hp2':'tanh'}     
    alg = LocalSearch(hp_space, hp_init)
    hp_next = alg.next(r)
    assert hp_next == hp_init, (hp_next, hp_init)

    # Add some data to results file.
    for i in range(3):
        hp      = {'hp1':i, 'hp2':'tanh'}
        expid   = r.on_start(hp=hp) # expid == i, the order in which we call on_start.
        metrics = {'loss':[0.1*i]*(i+1), 'val_loss':[0.2*i]*(i+1)}
        metricsfile = './temp/{}.pkl'.format(i)
        pkl.dump(metrics, open(metricsfile, 'wb'))
        r.on_finish(expid=expid, metricsfile=metricsfile)

    # Alg shouldn't suggest things already in results table.
    hp_space = {'hp1':[0,1,2,3],
                'hp2':['tanh']}
    hp_init = {'hp1':0, 'hp2':'tanh'}     
    alg = LocalSearch(hp_space, hp_init)
    hp_next = alg.next(r)
    assert hp_next == {'hp1':3, 'hp2':'tanh'}, (hp_next, hp_init)
    # Should return same suggestion if called again with same results.
    hp_next = alg.next(r)
    assert hp_next == {'hp1':3, 'hp2':'tanh'}, (hp_next, hp_init)
    # Should ignore the addition of combinations not in the hp_space.
    expid = r.on_start(hp={'hp1':3, 'hp2':'relu'}) # This one will be pending.
    metrics = {'loss':[0.1], 'val_loss':[0.2]}
    metricsfile = './temp/{}.pkl'.format(expid)
    pkl.dump(metrics, open(metricsfile, 'wb'))
    r.on_finish(expid=expid, metricsfile=metricsfile)
    hp_next = alg.next(r)
    assert hp_next == {'hp1':3, 'hp2':'tanh'}, (hp_next)
    # If the suggestion is pending, don't suggest it again.
    expid = r.on_start(hp={'hp1':3, 'hp2':'tanh'})
    hp_next = alg.next(r)
    assert hp_next in ['wait', 'stop'], (hp_next)
    metrics = {'loss':[0.1], 'val_loss':[0.2]}
    metricsfile = './temp/{}.pkl'.format(expid)
    pkl.dump(metrics, open(metricsfile, 'wb'))
    r.on_finish(expid=expid, metricsfile=metricsfile)
    hp_next = alg.next(r)
    assert hp_next == 'stop', (hp_next)

    # Expand hp2 space.
    hp_space = {'hp1':[0,1,2],
                'hp2':['tanh','relu']}
    alg = LocalSearch(hp_space)
    hp_next = alg.next(r)
    assert hp_next == {'hp1':0, 'hp2':'relu'}, hp_next
    # Let this exp have same as best loss.
    expid = r.on_start(hp=hp_next)
    metrics = {'loss':[0.0], 'val_loss':[0.0]} 
    metricsfile = './temp/{}.pkl'.format(expid)
    pkl.dump(metrics, open(metricsfile, 'wb'))
    r.on_finish(expid=expid, metricsfile=metricsfile)
    
    hp_next = alg.next(r)
    

    #assert hp_next['hp1'] == 0, hp_next
    #assert hp_next['hp2'] == 'tanh', hp_next
    #assert hp_next['hp3'] in [0,1,2], hp_next
    

  

    # Add some pending results.
    r.on_start(hp={'hp1':3, 'hp2':'relu'}) # This one will be pending.
    r.on_start(hp={'hp1':4, 'hp2':'relu'}) # This one will be pending.
 


if __name__ == '__main__':
    test_LocalSearch()
