from __future__ import absolute_import
import sherpa.hyperparameters as hyperparameters
import numpy as np
import math
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
import itertools
import abc

class AbstractSampler(object):
    def __init__(self, hplist=[]):
        self.hplist = hplist

    @abc.abstractmethod
    def next(self):
        ''' 
        Returns a dictionary where d[hp_name] = hp_sample.
        OR
        StopIterat
        '''
        pass

class RandomSampler(AbstractSampler):
    """
    Samples each hyperparameter independently.
    """
    def __init__(self, hplist=[]):
        for param in hplist:
            assert isinstance(param, hyperparameters.AbstractSampleableHyperparameter)
        super(RandomSampler, self).__init__(hplist=hplist)

    def next(self):
        ''' Returns a dictionary of d[hp_name] = hp_sample '''
        return {param.name: param.get_sample() for param in self.hplist}

class IterateSampler(AbstractSampler):
    def __init__(self, hp_combos):
        ''' hp_combos is a list of hp dictionaries.'''
        if not isinstance(hp_combos, list):
            raise ValueError()
        self.hp_combos = hp_combos
        self.counter = 0

    def next(self):
        if self.counter == len(self.hp_combos):
            raise StopIteration
        rval = self.hp_combos[self.counter]
        self.counter += 1
        return rval
             

class GridSearch(AbstractSampler):
    """
    Generate reasonable grid of hyperparameters based on bounded .
    
    INCOMPLETE:
    This is a partial solution that simply iterates over the different 
    combinations of the choice hyperparameters. The other parameters 
    are sampled as usual. This is because to build a grid, one must
    know the total number of models in advance.
    """
    def __init__(self, hplist=[]):
        for param in hplist:
            #assert isinstance(param, hyperparameters.BoundedDistributionHyperparameter)
            assert isinstance(param, hyperparameters.AbstractSampleableHyperparameter)
        super(GridSearch, self).__init__(hplist=hplist)

        # Define a stateful iterator.
        def griditer(hplist):
            # Iterate through discrete choices in order, but sample from distributions.
            # TODO: Compute grid choices for continuous distributions.
            choices  = {p.name: p.get_grid(k=None) for p in hplist if p.is_choice()}
            for ctuple in itertools.product(*choices.values()):
                # Sequential sample from choices.
                temp = dict(zip(choices.keys(), ctuple)) 
                # Independent sample.
                sample = {p.name: p.get_sample() for p in hplist if not p.is_choice()}
                sample.update(temp)
                yield sample
            raise StopIteration
        
        self.griditer = griditer(self.hplist)
    
    def next(self):
        ''' Returns a dictionary of d[hp_name] = hp_sample '''
        sample = next(self.griditer) # May throw StopIteration
        return sample

class LatinHypercube(AbstractSampler):
    """
    NEEDS ATTENTION
    Generates random hyperparameters based on parameter ranges
    """
    def __init__(self, param_ranges):
        self.param_ranges = param_ranges
        self.previous_hp = []

    def next(self):
        """
        Returns a dictionary of d[hp_name] = hp_sample
        """
        hp = {param.name: sample_from(param.distribution,
                                           param.dist_args) for param in self.param_ranges}
        while hp in self.previous_hp:
            hp = {param.name: sample_from(param.distribution,
                                               param.dist_args) for param in
                       self.param_ranges}

        self.previous_hp.append(hp)
        return hp

    def grow(self, hp, amount):
        for param_range in self.param_ranges:
            assert isinstance(param_range, GrowingHyperparameter)
            param_range.grow(value=hp[param_range.name],
                             amount=amount)
            print("{}: {}".format(param_range.name, param_range.weights))



class GaussianProcessEI(AbstractSampler):
    '''
     
    '''
    def __init__(self,num_eval_points=11):
        self.lower_is_better = True
        self.best_y = np.inf if self.lower_is_better else -np.inf
        self.num_eval_points = num_eval_points
        self.index = 0  
    def next(self,X,y):
        gp = GaussianProcessRegressor()
        X = np.atleast_2d(X).T
        gp.fit(X=X, y=y)
        X_candidates = []
        expected_improvement = []
        for i in range(1000):
            HP_star = random.uniform(0,0.1) #self.random_generator.next() hardcoded for now exploring lr (0, 0.1)
            X_star = np.array(HP_star)
            # print(X_star)
            y_star, y_star_sd = gp.predict(X_star, return_std=True) # predict the value using GP 
            ei = self.get_expected_improvement(y_star, y_star_sd) # can be modified to any acquisition function
            X_candidates.append(X_star)
            expected_improvement.append(ei)

        expected_improvement = np.array(expected_improvement)
        next_X = X_candidates[np.argmax(expected_improvement)]

        next_hp = next_X #self.turn_array_to_hparam_dict(next_X, X)
        print('next point we tried', next_hp)

        return next_hp


    def get_expected_improvement(self, y, y_std, epsilon=0.00001):
        with np.errstate(divide='ignore'):
            scaling_factor = (-1) ** self.lower_is_better
            z = scaling_factor * (y - self.best_y - epsilon)/y_std
            expected_improvement = scaling_factor * (y - self.best_y -
                                                     epsilon)*norm.cdf(z)
        return expected_improvement

