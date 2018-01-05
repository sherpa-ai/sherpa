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

    def get_suggestion(self, parameters, results, lower_is_better):
        if self.count >= self.max_num_trials:
            return None
        else:
            self.count += 1
            return {p.name: p.sample() for p in parameters}


class GridSearch(Algorithm):
    def __init__(self):
        self.count = 0
        self.grid = None

    def get_suggestion(self, parameters, results, lower_is_better):
        assert all(isinstance(p, Choice) for p in parameters), "Only Choice Parameters can be used with GridSearch"
        if self.count == 0:
            param_dict = {p.name: p.range for p in parameters}
            self.grid = list(sklearn.model_selection.ParameterGrid(param_dict))
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


def get_sample_results_and_params():
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