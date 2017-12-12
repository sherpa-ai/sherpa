import numpy
import pandas
import collections


class Trial(object):
    """
    A Trial object characterizes one set of parameters and an ID.

    # Attributes
        id (int): the Trial ID.
        parameters (dict): parameter-name, parameter-value pairs.
    """
    def __init__(self, id, parameters):
        self.id = id
        self.parameters = parameters


class Study(object):
    """
    A Study defines an entire optimization and its results.

    # Attributes:
        algorithm (sherpa.algorithms.Algorithm): takes results table and returns
            parameter set.
        results (pandas.DataFrame): contains results from this study.
        parameters (list[sherpa.Parameter]): parameters being optimized.
        stopping_rule (sherpa.stopping_rules.StoppingRule): rule for stopping
            trials prematurely.
        lower_is_better (bool): whether lower objective values are better.

    """
    def __init__(self, parameters, algorithm, stopping_rule, lower_is_better):
        self.parameters = parameters
        self.algorithm = algorithm
        self.stopping_rule = stopping_rule
        self.lower_is_better = lower_is_better
        self.results = pandas.DataFrame()

    def add_observation(self, trial, iteration, objective, context):
        """
        # Arguments:
            trial (sherpa.Trial): trial for which an observation is to be added.
            iteration (int): iteration number e.g. epoch.
            objective (float): objective value.
            context (dict): other metrics.
        """
        row = [
            ('Trial-ID', trial.id),
            ('Status', 'INTERMEDIATE'),
            ('Iteration', iteration)
        ]

        # Add parameters in sorted order
        p = trial.parameters
        row += sorted(p.items(), key=lambda t: t[0])

        # Add objective and sorted context
        row += [('Objective', objective)]
        row += sorted(context.items(), key=lambda t: t[0])

        # Use ordered dict to maintain order
        row = collections.OrderedDict([(key, [value]) for key, value in row])
        self.results = self.results.append(pandas.DataFrame.from_dict(row))

    def finalize(self, trial):
        """
        # Arguments:
            trial (sherpa.Trial): trial that is completed.
        """
        rows = self.results.loc[self.results['Trial-ID'] == trial.id]

        # Find best row as minimum or maximum objective
        best_idx = (rows['Objective'].idxmin() if self.lower_is_better
                    else rows['Objective'].idxmax())
        best_row = rows[best_idx]

        # Set status and append
        best_row['Status'] = 'COMPLETED'
        self.results = self.results.append(best_row, ignore_index=True)

    def get_suggestion(self):
        """
        # Returns:
            (dict) a parameter suggestion.
        """
        return self.algorithm.get_suggestion(self.parameters, self.results)

    def should_trial_stop(self, trial):
        """
        # Arguments:
            trial (sherpa.Trial): trial to be evaluated.

        # Returns:
            (bool) decision.
        """
        return self.stopping_rule.should_trial_stop(trial, self.results)


class Runner(object):
    """
    A class that runs a study with a scheduler and database.

    # Attributes:
        study (sherpa.Study): the study that is run.
        scheduler (sherpa.schedulers.Scheduler): a scheduler.
        database (sherpa.database.Database): the database.
    """
    def __init__(self, study, scheduler, database):
        pass

    def update_results(self):
        """
        Update results in study with any new results in database.
        """

    def update_jobs(self):
        """
        Get status of all running jobs from scheduler and update jobs.
        """

    def run_loop(self):
        """
        Run the optimization.
        """


class Parameter(object):
    """
    Base class for a parameter.
    """
    @staticmethod
    def from_config(config):
        if config.get('type') == 'continuous':
            return Continuous(name=config.get('name'),
                              range=config.get('range'),
                              scale=config.get('scale', 'linear'))
        elif config.get('type') == 'discrete':
            return Discrete(name=config.get('name'),
                            range=config.get('range'),
                            scale=config.get('scale', 'linear'))
        elif config.get('type') == 'choice':
            return Discrete(name=config.get('name'),
                            range=config.get('range'))
        else:
            raise ValueError("Got unexpected value for type: {}".format(
                config.get('type')))


class Continuous(Parameter):
    """
    Continuous parameter class.
    """
    def __init__(self, name, range, scale):
        self.name = name
        self.range = range
        self.scale = scale

    def sample(self):
        if self.scale == 'log':
            return 10**numpy.random.uniform(low=numpy.log10(range[0]),
                                            high=numpy.log10(range[1]))
        else:
            return numpy.random.uniform(low=range[0], high=range[1])


class Discrete(Parameter):
    """
    Discrete parameter class.
    """
    def __init__(self, name, range, scale):
        self.name = name
        self.range = range
        self.scale = scale

    def sample(self):
        if self.scale == 'log':
            return 10**numpy.random.randint(low=numpy.log10(range[0]),
                                            high=numpy.log10(range[1]))
        else:
            return numpy.random.randint(low=range[0], high=range[1])


class Choice(Parameter):
    """
    Choice parameter class.
    """
    def __init__(self, name, range):
        self.name = name
        self.range = range

    def sample(self):
        i = numpy.random.randint(low=0, high=len(self.range))
        return self.range[i]
