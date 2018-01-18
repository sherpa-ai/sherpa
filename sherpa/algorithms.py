import os
import numpy
import logging
import pandas
from .core import Choice, Continuous
import sklearn.model_selection


logging.basicConfig(level=logging.DEBUG)
alglogger = logging.getLogger(__name__)


class Algorithm(object):
    """
    Abstract algorithm that returns next parameters conditional on parameter
    ranges and previous results.
    """
    def get_suggestion(self, parameters, results, lower_is_better):
        """
        Returns a suggestion for parameter values based on results.

        # Arguments:
            parameters (list[sherpa.Parameter]): the parameters.
            results (pandas.DataFrame): all results so far.

        # Returns:
            (dict) of parameter values.
        """
        raise NotImplementedError("Algorithm class is not usable itself.")


class RandomSearch(Algorithm):
    """
    Regular Random Search.

    Expects to set a number of trials to yield.
    """
    def __init__(self, max_num_trials):
        self.max_num_trials = max_num_trials
        self.count = 0

    def get_suggestion(self, parameters, results=None, lower_is_better=True):
        if self.count >= self.max_num_trials:
            return None
        else:
            self.count += 1
            return {p.name: p.sample() for p in parameters}


class GridSearch(Algorithm):
    """
    Regular Grid Search. Expects Choice parameters.
    
    Example:
    ```
    hp_space = {'act': ['tanh', 'relu'],
                'lrinit': [0.1, 0.01],
                }
    parameters = sherpa.Parameter.grid(hp_space)
    alg = sherpa.algorithms.GridSearch()
    ```
    """
    def __init__(self):
        self.count = 0
        self.grid = None

    def get_suggestion(self, parameters, results=None, lower_is_better=True):
        assert all(isinstance(p, Choice) for p in parameters), "Only Choice Parameters can be used with GridSearch"
        if self.count == 0:
            param_dict = {p.name: p.range for p in parameters}
            self.grid = list(sklearn.model_selection.ParameterGrid(param_dict))
        if self.count == len(self.grid):
            return None
        else:
            params = self.grid[self.count]
            self.count += 1
            return params


class LocalSearch(Algorithm):
    """
    Local Search by Peter with perturbation modified
    """
    def __init__(self, num_random_seeds=10, seed_configurations=[]):
        # num_random_seeds + len(seed_configurations) needs to be larger than max_concurrent
        self.num_random_seeds = num_random_seeds
        self.seed_configurations = seed_configurations
        self.count = 0
        self.random_sampler = RandomSearch(self.num_random_seeds)

    def get_suggestion(self, parameters, results, lower_is_better):
        self.count += 1
        if self.count <= len(self.seed_configurations) + self.num_random_seeds:
            if len(self.seed_configurations) >= self.count:
                return self.seed_configurations[self.count-1]
            else:
                return self.random_sampler.get_suggestion(parameters, results,
                                                          lower_is_better)

        # Get best result so far
        try:
            best_idx = (results.loc[:, 'Objective'].argmin() if lower_is_better
                        else results.loc[:, 'Objective'].argmax())
        except ValueError:
            return self.random_sampler.get_suggestion(parameters,
                                                      results, lower_is_better)

        parameter_names = [p.name for p in parameters]
        best_params = results.loc[best_idx,
                                  parameter_names].to_dict()
        new_params = best_params
        # randomly choose one of the parameters and perturb it
        # while parameter in existing results
        # choose one dimension randomly and resample it
        alglogger.debug(new_params)
        while results.loc[:, parameter_names].isin({key: [value] for key, value in new_params.items()}).apply(all, axis=1).any():
            new_params = best_params.copy()
            p = numpy.random.choice(list(parameters))
            new_params[p.name] = p.sample()
            alglogger.debug(new_params)

        return new_params


class StoppingRule(object):
    """
    Abstract class to evaluate whether a trial should stop conditional on all
    results so far.
    """
    def should_trial_stop(self, trial, results, lower_is_better):
        """
        # Arguments:
            trial (sherpa.Trial): trial to be stopped.
            results (pandas.DataFrame): all results so far.
            lower_is_better (bool): whether lower objective values are better.

        # Returns:
            (bool) decision.
        """
        raise NotImplementedError("StoppingRule class is not usable itself.")


class MedianStoppingRule(StoppingRule):
    """
    Median Stopping-Rule similar to Golovin et al.
    "Google Vizier: A Service for Black-Box Optimization".
    
    # Description:
    For a Trial `t`, the best objective value is found.

    Then the best objective value for every other trial is found.

    Finally, the best-objective for the trial is compared to
    the median of the best-objectives of all other trials.

    If trial `t`'s best objective is worse than that median, it is
    stopped.

    If `t` has not reached the minimum iterations or there are not
    yet the requested number of comparison trials, `t` is not
    stopped.
    """
    def __init__(self, min_iterations=0, min_trials=1):
        self.min_iterations = min_iterations
        self.min_trials = min_trials

    def should_trial_stop(self, trial, results, lower_is_better):
        """
        # Arguments:
            trial (sherpa.Trial): trial to be stopped.
            results (pandas.DataFrame): all results so far.
            lower_is_better (bool): whether lower objective values are better.

        # Returns:
            (bool) decision.
        """
        if len(results) == 0:
            return False
        
        trial_rows = results.loc[results['Trial-ID'] == trial.id]
        
        max_iteration = trial_rows['Iteration'].max()
        if max_iteration < self.min_iterations:
            return False
        
        trial_obj_val = trial_rows['Objective'].min() if lower_is_better else trial_rows['Objective'].max()
        if numpy.isnan(trial_obj_val) and not trial_rows.empty:
            alglogger.debug("Value {} is NaN: {}, trial rows: {}".format(trial_obj_val, numpy.isnan(trial_obj_val), trial_rows))
            return True

        other_trial_ids = set(results['Trial-ID']) - {trial.id}
        comparison_vals = []

        for tid in other_trial_ids:
            trial_rows = results.loc[results['Trial-ID'] == tid]
            
            max_iteration = trial_rows['Iteration'].max()
            if max_iteration < self.min_iterations:
                continue

            valid_rows = trial_rows.loc[trial_rows['Iteration'] <= max_iteration]
            obj_val = valid_rows['Objective'].min() if lower_is_better else valid_rows['Objective'].max()
            comparison_vals.append(obj_val)

        if len(comparison_vals) < self.min_trials:
            return False

        if lower_is_better:
            decision = trial_obj_val > numpy.nanmedian(comparison_vals)
        else:
            decision = trial_obj_val < numpy.nanmedian(comparison_vals)

        return decision


def get_sample_results_and_params():
    """
    Call as
    parameters, results, lower_is_better = get_sample_results_and_params()
    to get a sample set of parameters, results and lower_is_better variable.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    results = pandas.read_csv(os.path.join(here, "sample_results.csv"), index_col=0)
    parameters = [Choice(name="param_a",
                         range=[1, 2, 3]),
                  Continuous(name="param_b",
                         range=[0, 1])]
    lower_is_better = True
    # return {'results': results, 'parameters': parameters,
    #         'lower_is_better': lower_is_better}
    return parameters, results, lower_is_better