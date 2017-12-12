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
    def should_trial_stop(self, trial, results):
        """
        # Arguments:
            trial (sherpa.Trial): trial to be stopped.

        # Returns:
            (bool) decision.
        """