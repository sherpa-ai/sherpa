from __future__ import print_function
from __future__ import absolute_import
import pytest
import numpy as np
from sherpa.hyperparameters import DistributionHyperparameter 
from sherpa.samplers import RandomSampler, GridSearch

def get_example_space():
    """
    Example hyperparameter space. List of Hyperparameter objects.
    """
    return [DistributionHyperparameter('learning_rate', 'log-uniform', (0.0001, 0.01)),
            DistributionHyperparameter('activation', 'choice', ['sigmoid', 'tanh', 'relu']),
            DistributionHyperparameter('dropout', 'uniform', {"low": 0., "high": 1.})]

@pytest.mark.run(order=1)
def test_RandomSampler():
    """
    Get example ranges, create RandomSampler, check that result
    is within ranges
    """
    hp_space = get_example_space()
    sampler = RandomSampler(hp_space)
    for _ in range(100):
        hp = sampler.next()
        assert 0.0001 <= hp['learning_rate'] <= 0.01
        assert hp['activation'] in ('sigmoid', 'tanh', 'relu')
        assert 0. <= hp['dropout'] <= 1.


if __name__ == '__main__':
    test_RandomSampler()
