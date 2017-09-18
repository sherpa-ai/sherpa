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
        ''' Returns a dictionary where d[hp_name] = hp_sample '''
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
    NEEDS ATTENTION
    '''
    def __init__(self, param_ranges, num_grid_points=11):
        super(self.__class__, self).__init__(param_ranges=param_ranges)
        self.random_generator = RandomGenerator(param_ranges=param_ranges)
        self.hparam_dict = {param.name: param for param in self.param_ranges}
        self.lower_is_better = True
        self.best_y = np.inf if self.lower_is_better else -np.inf
        self.num_grid_points = num_grid_points
        self.explored_idxs = set()
        self.old_hparam_idx = None

    # def next(self, X, y):
    #     if len(y) <= len(self.hparam_dict):
    #         return self.random_generator.next()
    #     # else:
    #     #     X_grid = self.get_grid(X, num_grid_points=3)
    #     # if len(y) <= X_grid.shape[0]:
    #     #     next_hparam_idx = len(y)-1
    #     #     return self.turn_array_to_hparam_dict(X_grid[next_hparam_idx], X)
    #     else:
    #         self.best_y = np.min(y) if self.lower_is_better else np.max(y)
    #
    #         gp = GaussianProcessRegressor()
    #         gp.fit(X=X, y=y)
    #         X_grid = self.get_grid(X, self.num_grid_points)
    #         y_grid, y_grid_std = gp.predict(X=X_grid, return_std=True)
    #
    #         # expected_improvement = [self.get_expected_improvement(y, y_std) for
    #         #                         y, y_std in zip(y_grid, y_grid_std)]
    #         expected_improvement_grid = self.get_expected_improvement(y_grid,
    #                                                              y_grid_std)
    #
    #         next_hparam_idx = expected_improvement_grid.argmax()
    #         # next_hparam_idx_cands = expected_improvement_grid.argsort()
    #         # for next_hparam_idx in np.flip(next_hparam_idx_cands, -1):
    #         #     if next_hparam_idx not in self.explored_idxs:
    #         #         self.explored_idxs.add(next_hparam_idx)
    #         #         break
    #         if next_hparam_idx == self.old_hparam_idx:
    #             return self.random_generator.next()
    #         else:
    #             self.old_hparam_idx = next_hparam_idx
    #             return self.turn_array_to_hparam_dict(X_grid[next_hparam_idx], X)

    def next(self, X, y):
        if len(y) < 2:
            return self.random_generator.next()
        else:
            self.best_y = np.min(y) if self.lower_is_better else np.max(y)

            gp = GaussianProcessRegressor()
            gp.fit(X=X, y=y)
            X_candidates = []
            expected_improvement = []
            for i in range(1000):
                HP_star = self.random_generator.next()
                X_star = np.array(self.turn_hparam_dict_to_array(HP_star))
                # print(X_star)
                y_star, y_star_sd = gp.predict(X_star, return_std=True)
                ei = self.get_expected_improvement(y_star, y_star_sd)
                X_candidates.append(X_star)
                expected_improvement.append(ei)

            # print(expected_improvement)
            expected_improvement = np.array(expected_improvement)
            # X_candidates = np.array(X_candidates)

            # print(np.argmax(expected_improvement))
            next_X = X_candidates[np.argmax(expected_improvement)]
            # print(next_X)

            next_hp = self.turn_array_to_hparam_dict(next_X, X)

            print(next_hp)

            return next_hp

    def get_grid(self, X, num_grid_points):
        """
        Strategy:
        If parameter name is in X.keys() it must be continuous, get its range (
        assert that distribution is uniform or log uniform)

        Else assume that the variable has a dummy variable so grid is [0, 1]


        Specifically:
        for each column in X make a linear space
            if column key is in parameter_ranges.names
                get its range
                build linspace
            else
                range must be [0, 1]

        Then make a list of those ranges
        ranges = [[0., 0.5, 1.], [0, 1], [0,1]]
        vals = np.meshgrid(*ranges)
        x_pred = np.vstack([val.ravel() for val in vals]).T
        """
        ranges = []
        for column in X:
            if column in self.hparam_dict:
                # for continuous variables
                hparam = self.hparam_dict[column]
                assert hparam.distribution == 'uniform' or \
                       hparam.distribution == 'log-uniform', "Must be uniform!"
                if isinstance(hparam.dist_args, dict):
                    low, high = hparam.dist_args['low'], hparam.dist_args['high']
                else:
                    low, high = hparam.dist_args
                ranges.append(np.linspace(low, high, num=num_grid_points))
            else:
                # for dummy columns i.e. discrete variables
                ranges.append([0, 1])

        vals = np.meshgrid(*ranges)
        X_grid = np.vstack([val.ravel() for val in vals]).T
        return X_grid

    def get_expected_improvement(self, y, y_std, epsilon=0.00001):
        with np.errstate(divide='ignore'):
            scaling_factor = (-1) ** self.lower_is_better
            z = scaling_factor * (y - self.best_y - epsilon)/y_std
            expected_improvement = scaling_factor * (y - self.best_y -
                                                     epsilon)*norm.cdf(z)
        return expected_improvement

    def turn_array_to_hparam_dict(self, X_array, X_df):
        hp = {}
        X_array = X_array.squeeze()
        for j, column in enumerate(X_df):
            if column in self.hparam_dict:
                hp[column] = X_array[j]
            else:
                if X_array[j] == 1:
                    val = column.split('_')[-1]
                    name = ''.join(column.split('_')[:-1])
                    hp[name] = val
        return hp

    def turn_hparam_dict_to_array(self, d, as_design_matrix=False):
        hparam_df = pd.DataFrame([d])
        return hparam_df if not as_design_matrix or hparam_df.empty else \
            pd.get_dummies(hparam_df, drop_first=True)


