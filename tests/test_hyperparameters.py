from __future__ import print_function
from __future__ import absolute_import
from sherpa.hyperparameters import DistributionHyperparameter

def test_choice():
    choices = [1, 'mystring', [2,3]] # Choices can be anything.
    p = DistributionHyperparameter(name='a', distribution='choice', dist_args=choices, seed=None)
    assert p.is_choice()
    assert p.num_choices() == 3
    for i in range(10):
        sample = p.get_sample()
        assert sample in choices, (sample, choices)
    for sample in p.get_grid():
        assert sample in choices, (sample, choices)

    choices = [1, 'mystring', [2,3]] # Choices can be anything.
    p = DistributionHyperparameter.fromlist(name='a', choices=choices)
    assert p.is_choice()
    assert p.num_choices() == 3
    assert p.get_sample() in choices
    for sample in p.get_grid():
        assert sample in choices

def test_uniform():
    low, high = -1, 3.0
    p = DistributionHyperparameter(name='a', distribution='uniform', dist_args={'low':low, 'high':high})
    assert not p.is_choice() 
    assert p.get_sample() >= low
    assert p.get_sample() <= high
    for sample in p.get_grid(k=10):
        assert (sample >= low) and (sample <= high)
        assert  low <= sample <= high

    low, high = .0001, 1.0
    p = DistributionHyperparameter(name='a', distribution='log-uniform', dist_args={'low':low, 'high':high})
    assert not p.is_choice() 
    assert p.get_sample() >= low
    assert p.get_sample() <= high
    for sample in p.get_grid(k=10):
        assert sample >= low and sample <= high

if __name__ == '__main__':
    test_choice()
    test_uniform()
