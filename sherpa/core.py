"""
SHERPA is a Python library for hyperparameter tuning of machine learning models.
Copyright (C) 2018  Lars Hertel, Peter Sadowski, and Julian Collado.

This file is part of SHERPA.

SHERPA is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

SHERPA is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with SHERPA.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import absolute_import
import os
import sys
import numpy
import pandas
import collections
import time
import logging
import socket
import multiprocessing
import warnings
import contextlib
import shlex
from .database import _Database
from .schedulers import _JobStatus
import datetime
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle


logger = logging.getLogger(__name__)
logging.getLogger('werkzeug').setLevel(level=logging.WARNING)
rng = numpy.random.RandomState(None)


class Trial(object):
    """
    Represents one parameter-configuration here referred to as one trial.

    Args:
        id (int): the Trial ID.
        parameters (dict): parameter-name, parameter-value pairs.
    """
    def __init__(self, id, parameters):
        self.id = id
        self.parameters = parameters


class TrialStatus(object):
    INTERMEDIATE = 'INTERMEDIATE'
    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'
    STOPPED = 'STOPPED'


class Study(object):
    """
    The core of an optimization.
    
    Includes functionality to get new suggested trials and add observations
    for those. Used internally but can also be used directly by the user.

    Args:
        parameters (list[sherpa.core.Parameter]): a list of parameter ranges.
        algorithm (sherpa.algorithms.Algorithm): the optimization algorithm.
        lower_is_better (bool): whether to minimize or maximize the objective.
        stopping_rule (sherpa.algorithms.StoppingRule): algorithm to stop badly
            performing trials.
        dashboard_port (int): the port for the dashboard web-server, if ``None``
            the first free port in the range `8880` to `9999` is found and used.
        disable_dashboard (bool): option to not run the dashboard.
        output_dir (str): directory path for CSV results.
        random_seed (int): seed to use for NumPy random number generators
            throughout.

    """
    def __init__(self,
                 parameters,
                 algorithm,
                 lower_is_better,
                 stopping_rule=None,
                 dashboard_port=None,
                 disable_dashboard=False,
                 output_dir=None):
        self.parameters = parameters
        self.algorithm = algorithm
        self.stopping_rule = stopping_rule
        self.lower_is_better = lower_is_better
        self.results = pandas.DataFrame()
        self.num_trials = 0
        self._trial_queue = collections.deque()
        self.output_dir = output_dir

        self._ids_to_stop = set()
        
        if not disable_dashboard:
            if sys.platform in ['cygwin', 'win32']:
                raise EnvironmentError('Dashboard not supported on Windows. Disable the dashboard and save the '
                                       'finalized study instead.')

            self._mgr = multiprocessing.Manager()
            self._results_channel = self._mgr.Namespace()
            self._results_channel.df = self.results
            self._stopping_channel = multiprocessing.Queue()
            dashboard_port = dashboard_port or _port_finder(8880, 9999)
            self.dashboard_process = self._run_web_server(dashboard_port)
        else:
            self.dashboard_process = None

    def add_observation(self, trial, objective, iteration=1, context={}):
        """
        Add a single observation of the objective value for a given trial.
        
        Args:
            trial (sherpa.core.Trial): trial for which an observation is to be
                added.
            iteration (int): iteration number e.g. epoch.
            objective (float): objective value.
            context (dict): other metrics or values to record.
        """
        assert isinstance(trial, Trial), "Trial must be sherpa.core.Trial"
        if not self.results.empty and\
                ((self.results['Trial-ID'] == trial.id)
                     & (self.results['Iteration'] == iteration)).any():
            raise ValueError("Observation for Trial-ID {} at Iteration {} "
                             "already exists.".format(trial.id, iteration))
        if not all(p.name in trial.parameters for p in self.parameters):
            raise ValueError("The trial is missing parameter entries. It "
                             "may not be from this study.")

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
            self._results_channel.df = self.results

    def finalize(self, trial, status='COMPLETED'):
        """
        Once a trial will not add any more observations it
        must be finalized with this function.
        
        Args:
            trial (sherpa.core.Trial): trial that is completed.
            status (str): one of 'COMPLETED', 'FAILED', 'STOPPED'.
        """
        assert isinstance(trial, Trial), "Trial must be sherpa.core.Trial"
        assert status in ['COMPLETED', 'FAILED', 'STOPPED']

        try:
            rows = self.results.loc[self.results['Trial-ID'] == trial.id]
            if len(rows) == 0:
                raise KeyError
        except KeyError:
            raise ValueError("Trial {} does not exist or did not "
                             "submit metrics.".format(trial.id))

        # Find best row as minimum or maximum objective
        best_idx = (rows['Objective'].idxmin() if self.lower_is_better
                    else rows['Objective'].idxmax())
        try:
            best_row = rows.loc[best_idx].copy()
        except TypeError:
            warnings.warn("Could not finalize trial {}. Only NaNs "
                          "encountered.".format(trial.id), RuntimeWarning)
            return

        # Set status and append
        best_row['Status'] = status
        best_row['Iteration'] = rows['Iteration'].max()
        self.results = self.results.append(best_row, ignore_index=True)

        if self.dashboard_process:
            self._results_channel.df = self.results

    def get_suggestion(self):
        """
        Obtain a new suggested trial.
        
        This function wraps the algorithm that was passed to the
        study.
        
        Returns:
            dict: a parameter suggestion.
        """
        if len(self._trial_queue) != 0:
            return self._trial_queue.popleft()
        
        p = self.algorithm.get_suggestion(self.parameters, self.results,
                                          self.lower_is_better)
        if isinstance(p, dict):
            self.num_trials += 1
            t = Trial(id=self.num_trials, parameters=p)
            return t
        else:
            return p

    def should_trial_stop(self, trial):
        """
        Determines whether given trial should stop.
        
        This function wraps the stopping rule provided to the
        study.
        
        Args:
            trial (sherpa.core.Trial): trial to be evaluated.

        Returns:
            bool: decision.
        """
        assert isinstance(trial, Trial), "Trial must be sherpa.core.Trial"
        if self.dashboard_process:
            while not self._stopping_channel.empty():
                self._ids_to_stop.add(self._stopping_channel.get())

        if trial.id in self._ids_to_stop:
            return True

        if self.stopping_rule:
            return self.stopping_rule.should_trial_stop(trial, self.results,
                                                        self.lower_is_better)
        else:
            return False
        
    def add_trial(self, trial):
        """
        Adds a trial into queue for next suggestion.
        
        Trials added via this method forego the suggestions
        made by the algorithm and are returned by the
        `get_suggestion` method on a first in first out
        basis.
        
        Args:
            trial (sherpa.core.Trial): the trial to be enqueued.
        """
        self._trial_queue.append(trial)

    def get_best_result(self):
        """
        Retrieve the best result so far.
        
        Returns:
            pandas.DataFrame: row of the best result.
        """
        if self.results.empty:
            return {}
        return self.algorithm.get_best_result(parameters=self.parameters,
                                              results=self.results,
                                              lower_is_better=
                                              self.lower_is_better)
        
    def _run_web_server(self, port):
        """
        Runs the SHERPA dashboard.

        Args:
            port (int): Port for web app.

        Returns:
            proc (multiprocessing.Process): the process that runs the web app.
            results_channel (multiprocessing.Queue): queue to put results in
            stopping_channel (multiprocessing.Queue): queue to get models to stop from.
        """
        from .app.app import app
        param_types = {}
        for p in self.parameters:
            if isinstance(p, Continuous) or (isinstance(p, Choice) and type(p.range[0])==float):
                param_types[p.name] = 'float'
            elif isinstance(p, Discrete) or (isinstance(p, Choice) and type(p.range[0])==int):
                param_types[p.name] = 'int'
            else:
                param_types[p.name] = 'string'
        app.parameter_types = param_types
                
        app.set_results_channel(self._results_channel)
        app.set_stopping_channel(self._stopping_channel)
        
        proc = multiprocessing.Process(target=app.run,
                                       kwargs={'port': port,
                                               'debug': True,
                                               'use_reloader': False,
                                               'host': '0.0.0.0',
                                               'threaded': True})
        msg = "\n" + "-"*55 + "\n"
        msg += "SHERPA Dashboard running. Access via\nhttp://{}:{} or " \
               "\nhttp://{}:{} if on a cluster, or " \
               "\nhttp://{}:{} if running locally.".format(
               socket.gethostbyname(socket.gethostname()), port,
               socket.gethostname(), port,
               "localhost", port)
        msg += "\n" + "-"*55
        logger.info(msg)
        
        proc.daemon = True
        proc.start()
        return proc

    def save(self, output_dir=None):
        """
        Stores results to CSV and attributes to config file.

        Args:
            output_dir (str): directory to store CSV to, only needed if Study
                output_dir is not defined.

        """
        if not output_dir:
            assert self.output_dir, "If no output-directory is specified, " \
                                    "a directory needs to be passed as argument"
        cfg = {'parameters': self.parameters,
               'lower_is_better': self.lower_is_better,
               'num_trials': self.num_trials}

        d = self.output_dir or output_dir
        with open(os.path.join(d, 'config.pkl'), 'wb') as f:
            pickle.dump(cfg, f)

        self.results.to_csv(os.path.join(self.output_dir or output_dir,
                                         'results.csv'), index=False)

    @staticmethod
    def load_dashboard(path):
        """
        Loads a study from an output dir without the algorithm.

        Args:
            path (str): the path to the output dir.

        Returns:
            sherpa.core.Study: the study running the dashboard, note that
                currently this study cannot be used to continue the optimization.
        """
        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            cfg = pickle.load(f)

        s = Study(parameters=cfg['parameters'],
                  lower_is_better=cfg['lower_is_better'],
                  algorithm=None, output_dir=path)

        results_path = os.path.join(path, 'results.csv')
        s.results = pandas.read_csv(results_path)
        s.num_trials = cfg['num_trials']
        s._results_channel.df = s.results
        return s

    def __iter__(self):
        """
        Allow to iterate over a study.
        """
        return self

    def __next__(self):
        """
        Allows to write `for trial in study:`.
        """
        t = self.get_suggestion()
        if isinstance(t, Trial):
            return t
        else:
            raise StopIteration
        
    def next(self):
        return self.__next__()

    def keras_callback(self, trial, objective_name, context_names=[]):
        """
        Keras Callbacks to add observations to study

        Args:
            trial (sherpa.core.Trial): trial to send metrics for.
            objective_name (str): the name of the objective e.g. ``loss``,
                ``val_loss``, or any of the submitted metrics.
            context_names (list[str]): names of all other metrics to be
                monitored.
        """
        import keras.callbacks
        send_call = lambda epoch, logs: self.add_observation(trial=trial,
                                                             iteration=epoch,
                                                             objective=logs[objective_name],
                                                             context={n: logs[n] for n in context_names})
        return keras.callbacks.LambdaCallback(on_epoch_end=send_call)


class _Runner(object):
    """
    Encapsulates all functionality needed to run a Study in parallel.

    Responsibilities:
    
    * Get rows from database and check if any new observations need to be added
        to ``Study``.
    * Update active trials, finalize any completed/stopped/failed trials.
    * Check what trials should be stopped and call scheduler ``kill_job``
        method.
    * Check if new trials need to be submitted, get parameters and submit as a
        job.

    Args:
        study (sherpa.core.Study): the study that is run.
        scheduler (sherpa.schedulers.Scheduler): a scheduler object.
        database (sherpa.database._Database): the database.
        max_concurrent (int): how many trials to run in parallel.
        command (list[str]): components of the command that runs a trial script
            e.g. ["python", "train_nn.py"].
        resubmit_failed_trials (bool): whether a failed trial should be
            resubmitted.
        
    """
    def __init__(self, study, scheduler, database, max_concurrent,
                 command, resubmit_failed_trials=False):
        self.max_concurrent = max_concurrent
        self.command = command
        self.resubmit_failed_trials = resubmit_failed_trials
        self.scheduler = scheduler
        self.database = database
        self.study = study

        self._done = False  # whether optimization is done.
        self._active_trials = []  # ids of trials that are active.
        self._queued_for_stopping = set()  # trials that need to be stopped.
        self._all_trials = {}  # maps trial id to Trial object, process ID.
        self._trial_status = {_JobStatus.finished: 'COMPLETED',
                              _JobStatus.killed: 'STOPPED',
                              _JobStatus.failed: 'FAILED',
                              _JobStatus.other: 'FAILED'}

    def update_results(self):
        """
        Get rows from database and check if anything new needs to be added to
        the results-table.
        """
        results = self.database.get_new_results()
        if results != [] and self._all_trials == {}:
            logger.warning(results)
            raise ValueError("Found unexpected results. Check the following\n"
                             "(1)\toutput_dir is empty\n"
                             "(2)\tno other database is running on this port.")

        for r in results:
            try:
                # Check if trial has already been collected.
                new_trial = (r.get('trial_id') not in
                             set(self.study.results['Trial-ID']))
            except KeyError:
                new_trial = True

            if not new_trial:
                trial_idxs = self.study.results['Trial-ID'] == r.get('trial_id')
                trial_rows = self.study.results[trial_idxs]
                new_observation = (r.get('iteration') not in
                                   set(trial_rows['Iteration']))
            else:
                new_observation = True

            if new_trial or new_observation:
                # Retrieve the Trial object
                tid = r.get('trial_id')
                tdict = self._all_trials[tid]
                t = tdict.get('trial')

                self.study.add_observation(trial=t,
                                           iteration=r.get('iteration'),
                                           objective=r.get('objective'),
                                           context=r.get('context'))

    def update_active_trials(self):
        """
        Update active trials, finalize any completed/stopped/failed trials.
        """
        for i in reversed(range(len(self._active_trials))):
            tid = self._active_trials[i]
            logger.debug('Updating active trials.')
            status = self.scheduler.get_status(self._all_trials[tid].get('job_id'))

            if status in [_JobStatus.finished, _JobStatus.failed,
                          _JobStatus.killed, _JobStatus.other]:

                if tid in self._queued_for_stopping:
                    self._queued_for_stopping.remove(tid)
                try:
                    self.study.finalize(trial=self._all_trials[tid].get('trial'),
                                        status=self._trial_status[status])
                    self.study.save()

                except ValueError as e:
                    warn_msg = str(e)
                    warn_msg += ("\nRelevant results not found in database."
                                 " Check whether:\n"
                                 "(1)\tTrial is submitting metrics via e.g. sherpa.Client.send_metrics()\n"
                                 "(2)\tTrial crashed\n"
                                 " Trial script output is in: ")
                    warn_msg += os.path.join(self.study.output_dir, 'jobs', 'trial_{}.out'.format(tid))
                    warnings.warn(warn_msg, RuntimeWarning)
                    if self.resubmit_failed_trials:
                        logger.info("Resubmitting Trial {}.".format(tid))
                        self.study.add_trial(self._all_trials[tid].get('trial'))
                self._active_trials.pop(i)

    def stop_bad_performers(self):
        """
        Check whether any of the running trials should stop and add them for
        stopping if necessary.
        """
        for tid in self._active_trials:
            if tid in self._queued_for_stopping:
                continue
            if self.study.should_trial_stop(self._all_trials[tid].get('trial')):
                logger.info("Stopping Trial {}".format(tid))
                self.scheduler.kill_job(self._all_trials[tid].get('job_id'))
                self._queued_for_stopping.add(tid)

    def submit_new_trials(self):
        """
        Get new trial and submit it to the job scheduler.
        """
        while len(self._active_trials) < self.max_concurrent:
            next_trial = self.study.get_suggestion()

            # Check if algorithm is done.
            if next_trial is None or next_trial == AlgorithmState.DONE:
                logger.info("Optimization Algorithm finished.")
                self._done = True
                break

            if next_trial == AlgorithmState.WAIT:
                break
            
            submit_msg = "\n" + "-"*55 + "\n" + "Submitting Trial {}:\n".format(next_trial.id)
            for pname, pval in next_trial.parameters.items():
                submit_msg += "\t{0:15}={1:>31}\n".format(str(pname), str(pval))
            submit_msg += "-"*55 + "\n"
            logger.info(submit_msg)

            self.database.enqueue_trial(next_trial)
            pid = self.scheduler.submit_job(command=self.command,
                                            env={'SHERPA_TRIAL_ID': str(next_trial.id),
                                                 'SHERPA_DB_HOST': socket.gethostname(),
                                                 'SHERPA_DB_PORT': str(self.database.port),
                                                 'SHERPA_OUTPUT_DIR': self.study.output_dir},
                                            job_name='trial_' + str(next_trial.id))
            self._all_trials[next_trial.id] = {'trial': next_trial,
                                              'job_id': pid}
            self._active_trials.append(next_trial.id)

    def run_loop(self):
        """
        Run the optimization loop.
        """
        while not self._done or self._active_trials:
            self.update_results()

            self.update_active_trials()

            self.stop_bad_performers()

            self.submit_new_trials()

            time.sleep(5)


def optimize(parameters, algorithm, lower_is_better,
             scheduler,
             command=None,
             filename=None,
             output_dir='./output_' + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
             max_concurrent=1,
             db_port=None, stopping_rule=None,
             dashboard_port=None, resubmit_failed_trials=False, verbose=1,
             load=False, mongodb_args={}, disable_dashboard=False):
    """
    Runs a Study with a scheduler and automatically runs a database in the
    background.

    Args:
        algorithm (sherpa.algorithms.Algorithm): takes results table and returns
            parameter set.
        parameters (list[sherpa.core.Parameter]): parameters being optimized.
        lower_is_better (bool): whether lower objective values are better.
        command (str): the command to run for the trial script.
        filename (str): the filename of the script to run. Will be run as
            "python <filename>".
        output_dir (str): where scheduler and database files will be stored.
        scheduler (sherpa.schedulers.Scheduler): a scheduler.
        max_concurrent (int): the number of trials that will be evaluated in
            parallel.
        db_port (int): port to run the database on.
        stopping_rule (sherpa.algorithms.StoppingRule): rule for stopping
            trials prematurely.
        dashboard_port (int): port to run the dashboard web-server on.
        resubmit_failed_trials (bool): whether to resubmit a trial if it failed.
        verbose (int, default=1): whether to print submit messages (0=no, 1=yes).
        load (bool): option to load study, currently not fully implemented.
        mongodb_args (dict[str, any]): arguments to MongoDB beyond port, dir,
            and log-path. Keys are the argument name without "--".
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not scheduler.output_dir:
        scheduler.output_dir = output_dir
        
    if verbose == 0:
        logger.setLevel(level=logging.INFO)
        logging.getLogger('dblogger').setLevel(level=logging.WARNING)

    study = Study(parameters=parameters,
                  algorithm=algorithm,
                  lower_is_better=lower_is_better,
                  stopping_rule=stopping_rule,
                  dashboard_port=dashboard_port,
                  output_dir=output_dir,
                  disable_dashboard=disable_dashboard)

    if command:
        runner_command = shlex.split(command)
    elif filename:
        runner_command = ['python', filename]
    else:
        raise ValueError("Need to provide either command or filename.")

    if load:
        study.load()

    if not db_port:
        db_port = _port_finder(27001, 27050)

    with _Database(db_dir=output_dir, port=db_port,
                   reinstantiated=load, mongodb_args=mongodb_args) as db:
        runner = _Runner(study=study,
                         scheduler=scheduler,
                         database=db,
                         max_concurrent=max_concurrent,
                         command=runner_command,
                         resubmit_failed_trials=resubmit_failed_trials)
        runner.run_loop()
    return study.get_best_result()


def run_dashboard(path):
    """
    Run the dashboard from a previously run optimization.

    Args:
        path (str): the output dir of the previous optimization.
    """
    s = Study.load_dashboard(path)


def _port_finder(start, end):
    """
    Helper function to find free port in range.
    
    Args:
        start (int): start point of port range.
        end (int): end point of port range.
    """
    def check_socket(host, port):
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            if sock.connect_ex((host, port)) == 0:
                return False
            else:
                return True
    try:
        hostname = socket.gethostname()
        for port in range(start, end):
            if check_socket(hostname, port):
                return port

    except socket.gaierror:
        raise BaseException('Hostname could not be resolved. Exiting')
        
    except socket.error:
        raise BaseException("Couldn't connect to server")


class Parameter(object):
    """
    Defines a hyperparameter with a name, type and associated range.

    Args:
        name (str): the parameter name.
        range (list): either ``[low, high]`` or ``[value1, value2, value3]``.
        scale (str): `linear` or `log`, defines sampling from linear or
            log-scale. Not defined for all parameter types.

    """
    def __init__(self, name, range):
        assert isinstance(name, str), "Parameter-Name needs to be a string."
        assert isinstance(range, list), "Parameter-Range needs to be a list."
        self.name = name
        self.range = range
        
    @staticmethod
    def from_dict(config):
        """
        Returns a parameter object according to the given dictionary config.

        Args:
            config (dict): parameter config.

        Example:
        ::

            {'name': '<name>',
             'type': '<continuous/discrete/choice>',
             'range': [<value1>, <value2>, ... ],
             'scale': <'log' to sample continuous/discrete from log-scale>}

        Returns:
            sherpa.core.Parameter: the parameter range object.

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

        Args:
            parameter_grid (dict): Dictionary mapping hyperparameter names 
                                   lists of possible values.

        Example:
            ::

                {'parameter_a': [aValue1, aValue2, ...],
                 'parameter_b': [bValue1, bValue2, ...],
                 ...}

        Returns:
            list[sherpa.core.Parameter]: list of parameter ranges for SHERPA.
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
        super(Continuous, self).__init__(name, range)
        self.scale = scale
        self.type = float
        if scale == 'log':
            assert all(r > 0. for r in range), "Range parameters must be " \
                                              "positive for log scale."

    def sample(self):
        try:
            if self.scale == 'log':
                return 10**rng.uniform(low=numpy.log10(self.range[0]),
                                                high=numpy.log10(self.range[1]))
            else:
                return rng.uniform(low=self.range[0], high=self.range[1])
        except ValueError as e:
            raise ValueError("{} causes error {}".format(self.name, e))


class Discrete(Parameter):
    """
    Discrete parameter class.
    """
    def __init__(self, name, range, scale='linear'):
        super(Discrete, self).__init__(name, range)
        self.scale = scale
        self.type = int
        if scale == 'log':
            assert all(r > 0 for r in range), "Range parameters must be " \
                                              "positive for log scale."

    def sample(self):
        try:
            if self.scale == 'log':
                return int(10**rng.uniform(low=numpy.log10(self.range[0]),
                                                high=numpy.log10(self.range[1])))
            else:
                return rng.randint(low=self.range[0], high=self.range[1])
        except ValueError as e:
            raise ValueError("{} causes error {}".format(self.name, e))


class Choice(Parameter):
    """
    Choice parameter class.
    """
    def __init__(self, name, range):
        super(Choice, self).__init__(name, range)
        self.type = type(self.range[0])

    def sample(self):
        i = rng.randint(low=0, high=len(self.range))
        return self.range[i]


class Ordinal(Parameter):
    """
    Ordinal parameter class. Categorical, ordered variable.
    """
    def __init__(self, name, range):
        super(Ordinal, self).__init__(name, range)
        self.type = type(self.range[0])

    def sample(self):
        i = rng.randint(low=0, high=len(self.range))
        return self.range[i]


class AlgorithmState(object):
    """
    Used internally to signal the sherpa._Runner class when to wait or when
    algorithm is done.
    """
    DONE = 'DONE'
    WAIT = 'WAIT'
