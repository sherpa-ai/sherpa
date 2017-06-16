from __future__ import absolute_import
from __future__ import division
from .resultstable import ResultsTable


class Hyperparameter(object):
    """
    A Hyperparameter instance captures the information about each of the hyperparameters to be optimized.
    
    # Arguments
        name: name of the hyperparameter. This will be used in the Model creation
        distr_args: List, Tuple or Dictionary, it used by the distribution function as arguments. 
                    In the default case of a Uniform distribution these refer to the minimum and maximum values from 
                    which to sample from. In general these are the  arguments taken as input by the corresponding numpy 
                    distribution function.
        distribution: String, name of the distribution to be used for sampling
                    the values. Must be numpy.random compatible. Exception is
                    'log-uniform' which samples uniformly between low and high
                    on a log-scale. Uniform distribution is used as
                    default.
    
    # Examples
        ```python
        Hyperparameter('learning_rate', distr_args=(0.0001, 0.1), distribution='log-uniform'),
        Hyperparameter('learning_rate', distr_args={low: 0.0001, high: 0.1}, distribution='uniform'),
        Hyperparameter('activation', distr_args=[('sigmoid', 'tanh', 'relu')], distribution='choice')
        ```
    
    """
    def __init__(self, name, distr_args, distribution='uniform'):
        assert isinstance(name, str), 'name should be a string, found {}'.format(str(name.__class__))
        assert (isinstance(distr_args, list) or isinstance(distr_args, tuple) or isinstance(distr_args, dict)), \
            'distr_args should be a dictionary, tuple or list, found {}'.format(str(distr_args.__class__))
        assert isinstance(distribution, str), 'distribution should be a string, found {}'.format(
            str(distribution.__class__))

        self.name = name
        self.distr_args = distr_args
        self.distribution = distribution
        return

class GrowingHyperparameter(object):
    def __init__(self, name, choices, start_value=0.5):
        self.distribution = 'choice'
        self.choices = choices
        self.name = name
        self.weights = [start_value] * len(self.choices)
        self.norm = lambda w: [v/sum(w) for v in w]

    def grow(self, value, amount):
        idx = self.choices.index(value)
        self.weights[idx] = max(self.weights[idx] + amount, 0)

    @property
    def distr_args(self):
        return {'a': self.choices, 'p': self.norm(self.weights)}


