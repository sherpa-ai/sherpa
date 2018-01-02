import numpy
from .core import Choice, logger
import sklearn.model_selection

class Algorithm(object):
    """
    Abstract algorithm that returns next parameters conditional on parameter
    ranges and previous results.
    """
    def get_suggestion(self, parameters, results):
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

    def get_suggestion(self, parameters, results):
        if self.count >= self.max_num_trials:
            return None
        else:
            self.count += 1
            return {p.name: p.sample() for p in parameters}


class GridSearch(Algorithm):
    def __init__(self):
        self.count = 0
        self.grid = None

    def get_suggestion(self, parameters, results):
        assert all(isinstance(p, Choice) for p in parameters), "Only Choice Parameters can be used with GridSearch"
        if self.count == 0:
            param_dict = {p.name: p.range for p in parameters}
            self.grid = list(sklearn.model_selection.ParameterGrid(param_dict))
        params = self.grid[self.count]
        self.count += 1
        return params


class LocalSearch(Algorithm):
    """
    Local Search by Peter
    """
    def __init__(self, first_setting=None, num_seed_trials=1):
        self.first_setting = first_setting
        self.random_sampler = RandomSearch(num_seed_trials)
        raise NotImplementedError

    def get_suggestion(self, parameters, results):
        if self.first_setting and len(results.get_matches(self.hp_init)) == 0:
            # Start this point first.
            _logger.info('Starting with {}'.format(self.hp_init))
            return self.hp_init


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
        trial_rows = results.loc[results['Trial-ID'] == trial.id]
        trial_rows_sorted = trial_rows.sort_values(by='Iteration')
        trial_obj_mean = trial_rows_sorted['Objective'].mean()
        max_iteration = trial_rows_sorted['Iteration'].max()
        if max_iteration < self.min_iterations:
            return False

        trial_ids = set(results['Trial-ID'])
        comparison_means = []

        for tid in trial_ids:
            if tid == trial.id:
                continue
            trial_rows = results.loc[results['Trial-ID'] == tid]

            valid_rows = trial_rows.loc[trial_rows['Iteration'] <= max_iteration]
            obj_mean = valid_rows['Objective'].mean()
            comparison_means.append(obj_mean)

        if len(comparison_means) < self.min_trials:
            return False

        if lower_is_better:
            decision = trial_obj_mean > numpy.median(comparison_means)
        else:
            decision = trial_obj_mean < numpy.median(comparison_means)

        return decision

