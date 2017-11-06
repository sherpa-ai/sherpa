from __future__ import print_function
from __future__ import absolute_import
import pytest
import numpy as np
import pickle as pkl
#from sherpa.hyperparameters import DistributionHyperparameter 
#from sherpa.samplers import RandomSampler, GridSearch
from sherpa.resultstable import ResultsTable

def get_example_space():
    """
    Example hyperparameter space. List of Hyperparameter objects.
    """
    return [DistributionHyperparameter('learning_rate', 'log-uniform', (0.0001, 0.01)),
            DistributionHyperparameter('activation', 'choice', ['sigmoid', 'tanh', 'relu']),
            DistributionHyperparameter('dropout', 'uniform', {"low": 0., "high": 1.})]

def simple_results_table():
    r = ResultsTable(dir='./temp', loss='loss', loss_summary=None, load_results=None)
    for i in range(3):
        hp      = {'hp1':i, 'hp2':'tanh'}
        expid   = r.on_start(hp=hp) # expid == i, the order in which we call on_start.
        metrics = {'loss':[0.1*i]*(i+1), 'val_loss':[0.2*i]*(i+1)}
        metricsfile = './temp/{}.pkl'.format(i)
        pkl.dump(metrics, open(metricsfile, 'wb'))
        r.on_finish(expid=expid, metricsfile=metricsfile)
    return r

@pytest.mark.run(order=1)
def test_get_best():
    """
    """
    r = simple_results_table()
 
    # Test get_k_lowest. 
    expids = r.get_k_lowest(ignore_pending=True, k=1)
    assert expids == [0]
    expids = r.get_k_lowest(ignore_pending=True, k=3)
    assert expids == [0, 1, 2]
    
    
    # Test get_best.
    hp_best = r.get_best(ignore_pending=True)
    assert hp_best == {'hp1':0, 'hp2':'tanh'}, hp_best
    hp_best = r.get_best(ignore_pending=True, k=1)
    assert hp_best == [{'hp1':0, 'hp2':'tanh'}], hp_best
    hp_best = r.get_best(ignore_pending=True, k=2)
    assert hp_best == [{'hp1':0, 'hp2':'tanh'}, {'hp1':1, 'hp2':'tanh'}], hp_best

@pytest.mark.run(order=1)
def test_get_matches():
    """
    """
    r = simple_results_table()
    
    hp = {'hp1':2, 'hp2':'tanh'}
    assert len(r.get_matches(hp)) == 1 
    assert r.get_matches(hp) == [2], r.get_matches(hp)
    
    hp = {'hp1':3, 'hp2':'tanh'}
    expid_1 = r.on_start(hp)
    assert len(r.get_matches(hp)) == 1, (r.get_matches(hp), hp)
    assert r.get_matches(hp) == [expid_1], r.get_matches(hp)
    expid_2 = r.on_start(hp)
    assert len(r.get_matches(hp)) == 2, (r.get_matches(hp), hp)
    assert set(r.get_matches(hp)) == set([expid_1, expid_2]), r.get_matches(hp)
    
    

if __name__ == '__main__':
    test_get_best()
    test_get_matches()
