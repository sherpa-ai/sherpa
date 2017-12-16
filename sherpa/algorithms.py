import numpy


class Algorithm(object):
    """
    Abstract algorithm that returns next parameters conditional on parameter
    ranges and previous results.
    """
    @staticmethod
    def get_suggestion(parameters, results):
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
    @staticmethod
    def get_suggestion(parameters, results):
        return {p.name: p.sample() for p in parameters}


class StoppingRule(object):
    """
    Abstract class to evaluate whether a trial should stop conditional on all
    results so far.
    """
    @staticmethod
    def should_trial_stop(trial, results, lower_is_better):
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
    @staticmethod
    def should_trial_stop(trial, results, lower_is_better):
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

        trial_ids = set(results['Trial-ID'])
        comparison_means = []

        for tid in trial_ids:
            if tid == trial.id:
                continue
            trial_rows = results.loc[results['Trial-ID'] == tid]
            valid_rows = trial_rows.loc[trial_rows['Iteration'] <= max_iteration]
            obj_mean = valid_rows['Objective'].mean()
            comparison_means.append(obj_mean)

        if lower_is_better:
            decision = trial_obj_mean > numpy.median(comparison_means)
        else:
            decision = trial_obj_mean < numpy.median(comparison_means)

        return decision

