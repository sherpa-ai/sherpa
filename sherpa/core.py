import numpy
import pandas
import collections
import time
import multiprocessing
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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
    def __init__(self, parameters, algorithm, stopping_rule, lower_is_better,
                 dashboard_port=None):
        self.parameters = parameters
        self.algorithm = algorithm
        self.stopping_rule = stopping_rule
        self.lower_is_better = lower_is_better
        self.results = pandas.DataFrame()
        self.num_trials = 0
        # if dashboard_port:
        #     self.dashboard_process = multiprocessing.Process(target=app.run,
        #                                                      kwargs={
        #                                                          'debug': True})
        #     self.dashboard_process.daemon = True
        #     self.dashboard_process.start()

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
        self.results = self.results.append(pandas.DataFrame.from_dict(row),
                                           ignore_index=True)

    def finalize(self, trial, status):
        """
        # Arguments:
            trial (sherpa.Trial): trial that is completed.
        """
        assert status in ['COMPLETED', 'FAILED', 'STOPPED']
        rows = self.results.loc[self.results['Trial-ID'] == trial.id]

        # Find best row as minimum or maximum objective
        best_idx = (rows['Objective'].idxmin() if self.lower_is_better
                    else rows['Objective'].idxmax())
        best_row = rows.ix[best_idx].copy()

        # Set status and append
        best_row['Status'] = status
        self.results = self.results.append(best_row, ignore_index=True)

    def get_suggestion(self):
        """
        # Returns:
            (dict) a parameter suggestion.
        """
        p = self.algorithm.get_suggestion(self.parameters, self.results)
        self.num_trials += 1
        t = Trial(id=self.num_trials, parameters=p)
        return t

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

    Responsibilities:
    -Get rows from Mongo DB and check if anything new needs to be added to table
    -Update active trials, finalize any completed/stopped/failed trials
    -Check what trials should be stopped and tell scheduler to stop those
    -Check if new trials need to be submitted, get parameters and submit

    # Attributes:
        study (sherpa.Study): the study that is run.
        scheduler (sherpa.schedulers.Scheduler): a scheduler.
        database (sherpa.database.Database): the database.
    """
    def __init__(self, study, scheduler, database, max_concurrent,
                 command, num_trials=10):
        self.max_concurrent = max_concurrent
        self.num_trials = num_trials
        self.command = command
        self.scheduler = scheduler
        self.database = database
        self.study = study
        self.active_trials = []  # ids of trials that are active
        self.all_trials = {}  # maps trial id to Trial object, process ID

    def update_results(self):
        """
        Get rows from Mongo DB and check if anything new needs to be added to
        the results-table.
        """
        results = self.database.get_results()

        for r in results:
            try:
                new_trial = r.get('trial_id') not in self.study.results['Trial-ID']
            except KeyError:
                new_trial = True

            if not new_trial:
                trial_idxs = self.study.results['Trial-ID'] == r.get('trial_id')
                trial_rows = self.study.results[trial_idxs]
                new_observation = r.iteration not in trial_rows['Iteration']
            else:
                new_observation = True

            if new_trial or new_observation:
                tid = r.get('trial_id')
                t = self.all_trials.get(tid).get('trial')
                self.study.add_observation(trial=t,
                                           iteration=r.get('iteration'),
                                           objective=r.get('objective'),
                                           context=r.get('context'))

    def update_active_trials(self):
        """
        Update active trials, finalize any completed/stopped/failed trials
        """
        for i in range(len(self.active_trials)-1, -1, -1):
            tid = self.active_trials[i]
            status = self.scheduler.get_status(self.all_trials[tid].get('job_id'))
            if status in ['COMPLETED', 'FAILED', 'STOPPED']:
                self.active_trials.pop(i)
                self.study.finalize(trial=self.all_trials[tid].get('trial'),
                                    status=status)

    def stop_bad_performers(self):
        for tid in self.active_trials:
            if self.study.should_trial_stop(self.all_trials[tid].get('trial')):
                logger.info("Stopping Trial {}".format(tid))
                self.scheduler.kill(self.all_trials[tid].get('job_id'))
                self.update_active_trials()

    def submit_new_trials(self):
        while len(self.active_trials) < self.max_concurrent:
            next_trial = self.study.get_suggestion()
            self.database.enqueue_trial(next_trial)
            pid = self.scheduler.submit(command=self.command)
            self.all_trials[next_trial.id] = {'trial': next_trial,
                                              'job_id': pid}
            self.active_trials.append(next_trial.id)

    def run_loop(self):
        """
        Run the optimization.
        """
        done = False
        while not done:
            self.update_results()

            self.update_active_trials()

            self.stop_bad_performers()

            self.submit_new_trials()

            time.sleep(10)

            if (self.num_trials == len(self.all_trials)
               and len(self.active_trials) == 0):
                done = True


class Parameter(object):
    """
    Base class for a parameter.
    """
    @staticmethod
    def from_dict(config):
        if config.get('type') == 'continuous':
            return Continuous(name=config.get('name'),
                              range=config.get('range'),
                              scale=config.get('scale', 'linear'))
        elif config.get('type') == 'discrete':
            return Discrete(name=config.get('name'),
                            range=config.get('range'),
                            scale=config.get('scale', 'linear'))
        elif config.get('type') == 'choice':
            return Choice(name=config.get('name'),
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
            return 10**numpy.random.uniform(low=numpy.log10(self.range[0]),
                                            high=numpy.log10(self.range[1]))
        else:
            return numpy.random.uniform(low=self.range[0], high=self.range[1])


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
            return 10**numpy.random.randint(low=numpy.log10(self.range[0]),
                                            high=numpy.log10(self.range[1]))
        else:
            return numpy.random.randint(low=self.range[0], high=self.range[1])


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
