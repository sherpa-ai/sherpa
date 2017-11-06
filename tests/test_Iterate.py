from __future__ import print_function
from __future__ import absolute_import
import pytest
import numpy as np
import pickle as pkl
#from sherpa.hyperparameters import DistributionHyperparameter 
#from sherpa.samplers import RandomSampler, GridSearch
from sherpa.resultstable import ResultsTable
from sherpa.algorithms import Iterate

@pytest.mark.run(order=1)
def test_Iterate():
    """
    """
    r = ResultsTable(dir='./temp', loss='loss', loss_summary=None, load_results=None)

    # Try on empty results table.
    hp_space = {'hp1':[0, 1],
                'hp2':['tanh'],
                'hp3':[[0,0], [1,1], [2,2]],
                'hp4':[{'hp4_0': 0}, {'hp4_1':1}]
                }
    alg = Iterate(hp_space)
    submitted = []
    for i in range(12): 
        hp_next = alg.next(r)
        assert type(hp_next) == dict, hp_next
        assert hp_next not in submitted
        submitted.append(hp_next)
    hp_next = alg.next(r)
    assert hp_next == 'stop'


if __name__ == '__main__':
    test_Iterate()
