from __future__ import absolute_import
from __future__ import division
import numpy as np
import math
import abc

class AbstractSampleableHyperparameter(object):
    """
    Abstract class for a Hyperparameter that is sampleable.
    """
    def __init__(self, name):
        if not isinstance(name, str):
            raise ValueError('Argument name should be a string, found {}'.format(str(name.__class__)))
        self.name = name
   
    def get_name(self):
        return self.name
 
    @abc.abstractmethod
    def get_sample(self):
        """Draw one sample."""
        raise NotImplementedError()

class AbstractGridHyperparameter(object):
    """
    Abstract class for a Hyperparameter that can return specified number of samples.
    """
    def __init__(self, name):
        if not isinstance(name, str):
            raise ValueError('Argument name should be a string, found {}'.format(str(name.__class__)))
        self.name = name
 
    def get_name(self):
        return self.name
    
    @abc.abstractmethod
    def get_grid(self, k):
        """ Return k samples."""
        raise NotImplementedError()


class DistributionHyperparameter(AbstractSampleableHyperparameter, AbstractGridHyperparameter):
    """
    Hyperparameter with a specified distribution.

    # Arguments
        name (str): The hyperparameter name.
        distribution (str, default='uniform'): Name of distribution as provided
            by numpy.random or "log-uniform".
        dist_args (list/dict): Distribution arguments as accepted by
            ```numpy.random.[distribution]```.
        seed (int, default=None): Seed for the random number generator.
    """
    def __init__(self, name, distribution='uniform', dist_args={}, seed=None):
        super(DistributionHyperparameter, self).__init__(name)
        if not isinstance(distribution, str):
            raise ValueError('distribution should be a string, found {}'.format(
                             str(distribution.__class__)))
        
        if not ((isinstance(dist_args, list) 
                or isinstance(dist_args, tuple) 
                or isinstance(dist_args, dict))):
            raise ValueError('dist_args should be a dictionary, \
                              tuple or list, found {}'.format(
                              str(dist_args.__class__)))
        self.distribution = distribution
        self.dist_args    = dist_args
        self.rng          = np.random.RandomState(seed)
        if not hasattr(self.rng, self.distribution):
            # Special cases need to be handled by get_sample.
            if self.distribution not in ['log-uniform']:
                AttributeError("Please choose an existing distribution from numpy.random.")
            

    @classmethod
    def fromlist(cls, name, choices):
        """Constructs hyperparameter from list of choices."""
        assert isinstance(choices, list) or isinstance(choices, tuple)
        return cls(name, distribution='choice', dist_args=choices)

    def is_choice(self):
        """ Is this a distribution over discrete choices? """
        return self.distribution == 'choice'
        
    def num_choices(self):
        """ Is this a distribution over discrete choices? """
        assert self.is_choice()
        return len(self.dist_args)

    def get_grid(self, k=None):
        """ Return k samples that are reasonably spaced out. """
        if (k is None) or (self.is_choice() and self.num_choices() == k):
            # Return all choices.
            assert self.is_choice()
            assert isinstance(self.dist_args, list)
            return self.dist_args
        elif self.is_choice():
            # Take first k choices.
            assert k <= self.num_choices(), 'Requested {}-grid over {} choices.'.format(k, self.num_choices())
            return self.dist_args[:k]
        else:
            # Return k random samples.
            # TODO: We should generate grid for certain distributions, based on cdf, quantiles.
            return [self.get_sample() for i in range(k)]  

    def get_sample(self):
        """ Draws one sample from distribution. """
        if self.distribution == 'log-uniform':
            if isinstance(self.dist_args, dict):
                low  = self.dist_args['low']
                high = self.dist_args['high']
            elif isinstance(self.dist_args, tuple):
                assert len(self.dist_args) == 2, self.dist_args
                low, high = self.dist_args
            else:
                raise ValueError('log-uniform takes two arguments: low and high')
            assert 0.0 < low, low
            assert 0.0 < high, high
            return 10 ** self.rng.uniform(math.log10(low), math.log10(high))
        elif self.distribution == 'choice':
            assert isinstance(self.dist_args, list) or isinstance(self.dist_args, tuple), self.dist_args
            #return self.rng.choice(self.dist_args) # Throws an error if dist_args contains list.
            l = len(self.dist_args)
            if l == 0:
                raise ValueError('No choices given!')
            elif l == 1:
                return self.dist_args[0]
            else:
                i = self.rng.randint(0, l)
                return self.dist_args[i]
            
        else:
            attr = getattr(self.rng, self.distribution)
            if isinstance(self.dist_args, dict):
                return attr(**self.dist_args)
            else:
                return attr(*self.dist_args)


class BoundedDistributionHyperparameter(DistributionHyperparameter):
    """
    Hyperparameter with specified, bounded distribution.
    Bounds are needed for grid searches.
    """
    def __init__(self, name, distribution='uniform', dist_args={}, seed=None):
        super(BoundedDistributionHyperparameter, self).__init__(name, distribution, dist_args, seed)
        
    def get_bounds(self):
        # Infers bounds.
        if self.name in ['uniform', 'log-uniform']:
            return (self.dist_args['low'], self.dist_args['high'])
        else:
            raise NotImplementedError('Unknown bounds for distribution: {}'.format(self.distribution))
