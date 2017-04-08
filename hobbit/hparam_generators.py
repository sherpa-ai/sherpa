import numpy as np
import math


def sample_from(dist, distr_args):
    """
    Draws one sample from given distribution with given args

    dist: Distribution to select values. This must be numpy.random compatible or 'log-uniform'
    distr_args: List containing the arguments for the distribution.
    
    
    """
    if dist == 'log-uniform':
        return 10 ** np.random.uniform(math.log10(distr_args[0]), math.log10(distr_args[1]))
    else:
        try:
            if isinstance(distr_args, dict):
                return eval('np.random.{}'.format(dist))(**distr_args)
            else:
                return eval('np.random.{}'.format(dist))(*distr_args)
        except AttributeError:
            AttributeError("Please choose an existing distribution from numpy.random and valid input parameters")

class RandomGenerator(object):
    """
    Generates random hyperparameters based on parameter ranges
    """
    def __init__(self, param_ranges):
        self.param_ranges = param_ranges

    def next(self):
        """
        Returns a dictionary of d[hp_name] = hp_sample
        """
        return {param.name: sample_from(param.distribution, param.distr_args) for param in self.param_ranges}
