import numpy as np
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm


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



class HyperparameterGenerator(object):
    def __init__(self, param_ranges):
        self.param_ranges = param_ranges

    def next(self):
        pass

class RandomGenerator(HyperparameterGenerator):
    """
    Generates random hyperparameters based on parameter ranges
    """
    def next(self):
        """
        Returns a dictionary of d[hp_name] = hp_sample
        """
        return {param.name: sample_from(param.distribution, param.distr_args) for param in self.param_ranges}


class GaussianProcessEI(HyperparameterGenerator):
    def __init__(self, param_ranges, num_grid_points=11):
        super(self.__class__, self).__init__(param_ranges=param_ranges)
        self.random_generator = RandomGenerator(param_ranges=param_ranges)
        self.hparam_dict = {param.name: param for param in self.param_ranges}
        self.lower_is_better = True
        self.best_y = np.inf if self.lower_is_better else -np.inf
        self.num_grid_points = num_grid_points
        self.explored_idxs = set()

    def next(self, X, y):
        if len(y) < 1:
            return self.random_generator.next()
        else:
            X_grid = self.get_grid(X, num_grid_points=3)
        if len(y) < X_grid.shape[0]:
            next_hparam_idx = len(y)
        else:
            self.best_y = np.min(y) if self.lower_is_better else np.max(y)

            gp = GaussianProcessRegressor()
            gp.fit(X=X, y=y)
            X_grid = self.get_grid(X, self.num_grid_points)
            y_grid, y_grid_std = gp.predict(X=X_grid, return_std=True)

            # expected_improvement = [self.get_expected_improvement(y, y_std) for
            #                         y, y_std in zip(y_grid, y_grid_std)]
            expected_improvement_grid = self.get_expected_improvement(y_grid,
                                                                 y_grid_std)

            next_hparam_idx_cands = expected_improvement_grid.argsort()
            for next_hparam_idx in np.flip(next_hparam_idx_cands, -1):
                if next_hparam_idx not in self.explored_idxs:
                    self.explored_idxs.add(next_hparam_idx)
                    break

        return self.turn_array_to_hparam_dict(X_grid[next_hparam_idx], X)

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
                if isinstance(hparam.distr_args, dict):
                    low, high = hparam.distr_args['low'], hparam.distr_args['high']
                else:
                    low, high = hparam.distr_args
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
        hparams = {}
        for j, column in enumerate(X_df):
            if column in self.hparam_dict:
                hparams[column] = X_array[j]
            else:
                if X_array[j] == 1:
                    val = column.split('_')[-1]
                    name = ''.join(column.split('_')[:-1])
                    hparams[name] = val
        return hparams


