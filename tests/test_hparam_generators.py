from __future__ import print_function
from __future__ import absolute_import
import pytest
from hobbit.core import Hyperparameter
from hobbit import hparam_generators


def get_example_hparam_ranges():
    """
    These are a set of hyper-parameter ranges for tests below.
    """
    return [Hyperparameter('learning_rate', (0.0001, 0.01), 'log-uniform'),
            Hyperparameter('activation', [['sigmoid', 'tanh', 'relu']], 'choice'),
            Hyperparameter('dropout', {"low": 0., "high": 1.}, 'uniform')]

@pytest.mark.run(order=2)
def test_sample_from():
    """
    Test the sample_from function
    """
    assert 0. <= hparam_generators.sample_from('uniform', (0., 1.)) <= 1.
    assert hparam_generators.sample_from('choice',
                                         [('A', 'B', 'C')]) in ('A', 'B', 'C')

@pytest.mark.run(order=1)
def test_random_generator():
    """
    Get example ranges, create RandomGenerator, check that result
    is within ranges
    """
    hparam_ranges = get_example_hparam_ranges()
    random_gen = hparam_generators.RandomGenerator(hparam_ranges)
    for _ in range(100):
        hparams = random_gen.next()
        assert 0.0001 <= hparams['learning_rate'] <= 0.01
        assert hparams['activation'] in ('sigmoid', 'tanh', 'relu')
        assert 0. <= hparams['dropout'] <= 1.

if __name__ == '__main__':
    pytest.main([__file__])
