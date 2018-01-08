from __future__ import absolute_import
import os
import numpy
import pandas
import collections
import time
import logging
import multiprocessing
import warnings
from .database import Database
from .schedulers import JobStatus

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
        dashboard_port (int): port to run the dashboard web-server on.

    """
    def __init__(self, parameters, algorithm, lower_is_better,
                 stopping_rule=None, dashboard_port=None):
        self.parameters = parameters
        self.algorithm = algorithm
        self.stopping_rule = stopping_rule
        self.lower_is_better = lower_is_better
        self.results = pandas.DataFrame()
        self.num_trials = 0

        self.ids_to_stop = set()
        
        if dashboard_port:
            # self.results_channel = multiprocessing.Value(pandas.DataFrame, self.results)
            self.mgr = multiprocessing.Manager()
            self.results_channel = self.mgr.Namespace()
            self.results_channel.df = self.results
            self.stopping_channel = multiprocessing.Queue()
            self.dashboard_process = self.run_web_server(dashboard_port)
        else:
            self.dashboard_process = None

    def add_observation(self, trial, iteration, objective, context={}):
        """
        # Arguments:
            trial (sherpa.Trial): trial for which an observation is to be added.
            iteration (int): iteration number e.g. epoch.
            objective (float): objective value.
            context (dict): other metrics.
        """
        assert isinstance(trial, Trial), "Trial must be sherpa.Trial"

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

        if self.dashboard_process:
            self.results_channel.df = self.results

    def finalize(self, trial, status='COMPLETED'):
        """
        # Arguments:
            trial (sherpa.Trial): trial that is completed.
        """
        assert isinstance(trial, Trial), "Trial must be sherpa.Trial"
        assert status in ['COMPLETED', 'FAILED', 'STOPPED']

        try:
            rows = self.results.loc[self.results['Trial-ID'] == trial.id]
            if len(rows) == 0:
                raise KeyError
        except KeyError:
            raise ValueError("Trial {} does not exist or did not submit metrics.".format(trial.id))
        # Find best row as minimum or maximum objective
        best_idx = (rows['Objective'].idxmin() if self.lower_is_better
                    else rows['Objective'].idxmax())
        best_row = rows.ix[best_idx].copy()

        # Set status and append
        best_row['Status'] = status
        self.results = self.results.append(best_row, ignore_index=True)

        if self.dashboard_process:
            self.results_channel.df = self.results

    def get_suggestion(self):
        """
        # Returns:
            (dict) a parameter suggestion.
        """
        p = self.algorithm.get_suggestion(self.parameters, self.results,
                                          self.lower_is_better)
        if not p:
            return None
        else:
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
        assert isinstance(trial, Trial), "Trial must be sherpa.Trial"
        if self.dashboard_process:
            while not self.stopping_channel.empty():
                self.ids_to_stop.add(self.stopping_channel.get())
        # logger.debug(self.ids_to_stop)
        if trial.id in self.ids_to_stop:
            return True

        if self.stopping_rule:
            return self.stopping_rule.should_trial_stop(trial, self.results,
                                                        self.lower_is_better)
        else:
            return False

    def get_best_result(self):
        # Get best result so far
        best_idx = (self.results.loc[:, 'Objective'].argmin() if self.lower_is_better
                    else self.results.loc[:, 'Objective'].argmax())

        best_result = self.results.loc[best_idx, :].to_dict()
        best_result.pop('Status')
        return best_result
        
    def run_web_server(self, port):
        """
        Runs the web server.

        # Arguments:
            port (int): Port for web app.

        # Returns:
            proc (multiprocessing.Process): the process that runs the web app.
            results_channel (multiprocessing.Queue): queue to put results in
            stopping_channel (multiprocessing.Queue): queue to get models to stop from.
        """
        from .app.app import app
        app.set_results_channel(self.results_channel)
        app.set_stopping_channel(self.stopping_channel)
        proc = multiprocessing.Process(target=app.run,
                                       kwargs={'port': port, 'debug': True, 'use_reloader': False, 'host': '', 'threaded': True})
        proc.daemon = True
        proc.start()
        return proc

    def __iter__(self):
        """
        Allow to iterate over a study.
        """
        return self

    def __next__(self):
        """
        Use study as a generator.
        """
        t = self.get_suggestion()
        if t is None:
            raise StopIteration
        else:
            return t
        
    def next(self):
        return self.__next__()


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
                 command):
        self.max_concurrent = max_concurrent
        self.command = command
        self.done = False
        self.scheduler = scheduler
        self.database = database
        self.study = study
        self.active_trials = []  # ids of trials that are active
        self.all_trials = {}  # maps trial id to Trial object, process ID
        self.trial_status = {JobStatus.finished: 'COMPLETED',
                             JobStatus.killed: 'STOPPED',
                             JobStatus.failed: 'FAILED',
                             JobStatus.other: 'FAILED'}

    def update_results(self):
        """
        Get rows from Mongo DB and check if anything new needs to be added to
        the results-table.
        """
        results = self.database.get_new_results()
        if results != [] and self.all_trials == {}:
            logger.debug(results)
            raise ValueError("Found unexpected results. Check the following\n"
                             "(1)\toutput_dir is empty\n"
                             "(2)\tno other database is running on this port.")

        for r in results:
            try:
                new_trial = r.get('trial_id') not in set(self.study.results['Trial-ID'])
            except KeyError:
                new_trial = True

            if not new_trial:
                trial_idxs = self.study.results['Trial-ID'] == r.get('trial_id')
                trial_rows = self.study.results[trial_idxs]
                new_observation = r.get('iteration') not in set(trial_rows['Iteration'])
            else:
                new_observation = True
            # logger.debug("Collected Result:\n\tTrial ID: {}\n\tIteration: {}"
            #              "\n\tNew Trial: {}\n\tNew Observation: {}"
            #              "".format(r.get('trial_id'), r.get('iteration'),
            #                        new_trial, new_observation))
            if new_trial or new_observation:
                tid = r.get('trial_id')
                tdict = self.all_trials[tid]
                t = tdict.get('trial')
                self.study.add_observation(trial=t,
                                           iteration=r.get('iteration'),
                                           objective=r.get('objective'),
                                           context=r.get('context'))

    def update_active_trials(self):
        """
        Update active trials, finalize any completed/stopped/failed trials
        """
        # logger.debug("Updating trials")
        for i in range(len(self.active_trials)-1, -1, -1):
            tid = self.active_trials[i]
            status = self.scheduler.get_status(self.all_trials[tid].get('job_id'))
            # logger.debug("Trial with ID {} has status {}".format(tid, status))
            if status in [JobStatus.finished, JobStatus.failed,
                          JobStatus.killed, JobStatus.other]:
                self.update_results()
                try:
                    self.study.finalize(trial=self.all_trials[tid].get('trial'),
                                        status=self.trial_status[status])
                except ValueError as e:
                    warnings.warn(str(e) + "\nRelevant results not found in database. Check that"
                                  " Client has correct host/port and is submitting"
                                  " metrics.", RuntimeWarning)
                self.active_trials.pop(i)

    def stop_bad_performers(self):
        for tid in self.active_trials:
            if self.study.should_trial_stop(self.all_trials[tid].get('trial')):
                logger.info("Stopping Trial {}".format(tid))
                self.scheduler.kill_job(self.all_trials[tid].get('job_id'))
                self.update_active_trials()

    def submit_new_trials(self):
        while len(self.active_trials) < self.max_concurrent:
            next_trial = self.study.get_suggestion()

            # Check if algorithm is done.
            if not next_trial:
                self.done = True
                break

            logger.info("Submitting Trial {} with parameters"
                         " {}".format(next_trial.id, next_trial.parameters))

            self.database.enqueue_trial(next_trial)
            pid = self.scheduler.submit_job(command=self.command)
            self.all_trials[next_trial.id] = {'trial': next_trial,
                                              'job_id': pid}
            self.active_trials.append(next_trial.id)

    def run_loop(self):
        """
        Run the optimization.
        """
        logger.debug("Running Loop")
        while not self.done or len(self.active_trials) != 0:
            self.update_results()

            self.update_active_trials()

            self.stop_bad_performers()

            self.submit_new_trials()

            logger.info("Best results so far:\n"
                        "{}".format(self.study.get_best_result()))

            # logger.info(self.study.results)
            time.sleep(1)


def optimize(parameters, algorithm, lower_is_better, filename, output_dir,
             scheduler, max_concurrent=1, db_port=27010, stopping_rule=None,
             dashboard_port=None):
    """
    Runs a Study with a scheduler and automatically runs a database in the
    background.

    # Arguments:
        algorithm (sherpa.algorithms.Algorithm): takes results table and returns
            parameter set.
        results (pandas.DataFrame): contains results from this study.
        parameters (list[sherpa.Parameter]): parameters being optimized.
        stopping_rule (sherpa.stopping_rules.StoppingRule): rule for stopping
            trials prematurely.
        lower_is_better (bool): whether lower objective values are better.
        dashboard_port (int): port to run the dashboard web-server on.
        filename (str): the name of the file which is called to evaluate
            configurations
        study (sherpa.Study): the Study to be run.
        output_dir (str): where scheduler and database files will be stored.
        scheduler (sherpa.Scheduler): a scheduler.
        max_concurrent (int): the number of trials that will be evaluated in
            parallel.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    study = Study(parameters=parameters,
                  algorithm=algorithm,
                  lower_is_better=lower_is_better,
                  stopping_rule=stopping_rule,
                  dashboard_port=dashboard_port)
    with Database(db_dir=output_dir, port=db_port) as db:
        runner = Runner(study=study,
                        scheduler=scheduler,
                        database=db,
                        max_concurrent=max_concurrent,
                        command=' '.join(['python', filename]))
        runner.run_loop()
    return study.get_best_result()


class Parameter(object):
    """
    Defines a hyperparameter with a name, type and associated range.
    """
    @staticmethod
    def from_dict(config):
        """
        Returns a parameter object according to the given dictionary config.

        # Arguments:
            config (dict): parameter config of the format
                {'name': '<the-name>',
                 'type': '<continuous/discrete/choice>',
                 'range': [value1, value2, ... ],
                 'scale': 'log' to sample continuous/discrete from log-scale}

        # Returns:
            (sherpa.Parameter)

        """
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

    @staticmethod
    def grid(parameter_grid):
        """
        Creates a list of parameters given a parameter grid.

        # Arguments:
            parameter_grid (dict): dictionary of the form
                {'parameter_a': [aValue1, aValue2, ...],
                 'parameter_b': [bValue1, bValue2, ...],
                 ...}

        # Returns:
            (list[sherpa.Parameter])
        """
        plist = []
        for pname, prange in parameter_grid.items():
            p = Parameter.from_dict({'name': pname,
                                     'type': 'choice',
                                     'range': prange})
            plist.append(p)
        return plist


class Continuous(Parameter):
    """
    Continuous parameter class.
    """
    def __init__(self, name, range, scale='linear'):
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
