from __future__ import absolute_import
import os
import numpy
import pandas
import collections
import time
import logging
from logging.handlers import RotatingFileHandler
import socket
import multiprocessing
import warnings
import contextlib
from .database import Database
from .schedulers import JobStatus

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger('werkzeug').setLevel(level=logging.WARNING)


class Trial(object):
    """
    Represents one parameter-configuration here referred to as one trial.

    # Attributes
        id (int): the Trial ID.
        parameters (dict): parameter-name, parameter-value pairs.
    """
    def __init__(self, id, parameters):
        self.id = id
        self.parameters = parameters


class Study(object):
    """
    The core of an optimization.
    
    Includes functionality to get new suggested trials and add observations
    for those. Used internally but can also be used directly by the user.

    # Attributes:
        algorithm (sherpa.algorithms.Algorithm): takes results table and returns
            parameter set.
        results (pandas.DataFrame): contains results from this study.
        parameters (list[sherpa.Parameter]): parameters being optimized.
        stopping_rule (sherpa.stopping_rules.StoppingRule): rule for stopping
            trials prematurely.
        lower_is_better (bool): whether lower objective values are better.
        dashboard_port (int): port to run the dashboard web-server on.
        disable_dashboard (bool): whether to not run the dashboard.
        output_dir (str): directory to store web-app output and results-CSV.

    """
    def __init__(self, parameters, algorithm, lower_is_better,
                 stopping_rule=None, dashboard_port=None,
                 disable_dashboard=False,
                 output_dir=None):
        """
        # Arguments:
            parameters (list[sherpa.Parameter]): a list of parameter ranges.
            algorithm (sherpa.algorithm): the optimization algorithm.
            lower_is_better (bool): whether to minimizer or maximize objective.
            stopping_rule (sherpa.StoppingRule): algorithm to stop badly
                performing trials.
            dashboard_port (int): the port for the dashboard web-server.
            disable_dashboard (bool): option to not run the dashboard.
            output_dir (str): directory path for CSV results.
        """
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
            self._mgr = multiprocessing.Manager()
            self._results_channel = self._mgr.Namespace()
            self._results_channel.df = self.results
            self._stopping_channel = multiprocessing.Queue()
            dashboard_port = dashboard_port or port_finder(8880, 9999)
            self.dashboard_process = self.run_web_server(dashboard_port)
        else:
            self.dashboard_process = None

    def add_observation(self, trial, iteration, objective, context={}):
        """
        Add a single observation of the objective value for a given trial.
        
        # Arguments:
            trial (sherpa.Trial): trial for which an observation is to be added.
            iteration (int): iteration number e.g. epoch.
            objective (float): objective value.
            context (dict): other metrics or values to record.
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
            self._results_channel.df = self.results

    def finalize(self, trial, status='COMPLETED'):
        """
        Once a trial will not add any more observations it
        must be finalized with this function.
        
        # Arguments:
            trial (sherpa.Trial): trial that is completed.
            status (str): one of 'COMPLETED', 'FAILED', 'STOPPED'.
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
        try:
            best_row = rows.ix[best_idx].copy()
        except TypeError:
            warnings.warn("Could not finalize trial {}. Only NaNs encountered.".format(trial.id), RuntimeWarning)
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
        
        # Returns:
            (dict) a parameter suggestion.
        """
        if len(self._trial_queue) != 0:
            return self._trial_queue.popleft()
        
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
        Determines whether given trial should stop.
        
        This function wraps the stopping rule provided to the
        study.
        
        # Arguments:
            trial (sherpa.Trial): trial to be evaluated.

        # Returns:
            (bool) decision.
        """
        assert isinstance(trial, Trial), "Trial must be sherpa.Trial"
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
        
        # Arguments:
            trial (sherpa.Trial): the trial to be enqueued.
        """
        self._trial_queue.append(trial)
        
    def to_csv(self, output_dir=None):
        """
        Stores results to CSV.
        
        # Arguments:
            output_dir (str): directory to store CSV to.
        """
        if not output_dir:
            assert self.output_dir, "If no output-directory is specified, a directory needs to be passed as argument"
        self.results.to_csv(os.path.join(self.output_dir or output_dir, 'results.csv'), index=False)

    def get_best_result(self):
        """
        Retrieve the best result so far.
        
        # Returns:
            pandas.DataFrame
        """
        # Get best result so far
        best_idx = (self.results.loc[:, 'Objective'].idxmin() if self.lower_is_better
                    else self.results.loc[:, 'Objective'].idxmax())

        best_result = self.results.loc[best_idx, :].to_dict()
        best_result.pop('Status')
        return best_result
        
    def run_web_server(self, port):
        """
        Runs the SHERPA dashboard.

        # Arguments:
            port (int): Port for web app.

        # Returns:
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
                                       kwargs={'port': port, 'debug': True, 'use_reloader': False, 'host': '', 'threaded': True})
        msg = "\n" + "-"*55 + "\n"
        msg += "SHERPA Dashboard running on http://{}:{}".format(socket.gethostbyname(socket.gethostname()), port)
        msg += "\n" + "-"*55
        logger.info(msg)
        
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
        Allows to write `for trial in study:`.
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
    Encapsulates all functionality needed to run a Study in parallel.

    Responsibilities:
    
    -Get rows from database and check if any new observations need to be added to study.
    -Update active trials, finalize any completed/stopped/failed trials.
    -Check what trials should be stopped and tell database to send stop signal.
    -Check if new trials need to be submitted, get parameters and submit as a job.

    # Attributes:
        study (sherpa.Study): the study that is run.
        scheduler (sherpa.schedulers.Scheduler): a scheduler.
        database (sherpa.database.Database): the database.
        max_concurrent (int): how many trials to run in parallel.
        command (str): the command that runs a trial script e.g. "python train_nn.py".
        resubmit_failed_trials (bool): whether a failed trial should be resubmitted.
        
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
        self._trial_status = {JobStatus.finished: 'COMPLETED',
                              JobStatus.killed: 'STOPPED',
                              JobStatus.failed: 'FAILED',
                              JobStatus.other: 'FAILED'}

    def update_results(self):
        """
        Get rows from database and check if anything new needs to be added to
        the results-table.
        """
        results = self.database.get_new_results()
        if results != [] and self._all_trials == {}:
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
        # logger.debug("Updating trials")
        for i in range(len(self._active_trials)-1, -1, -1):
            tid = self._active_trials[i]
            status = self.scheduler.get_status(self._all_trials[tid].get('job_id'))
            # logger.debug("Trial with ID {} has status {}".format(tid, status))
            if status in [JobStatus.finished, JobStatus.failed,
                          JobStatus.killed, JobStatus.other]:
                # self.update_results()
                if tid in self._queued_for_stopping:
                    self._queued_for_stopping.remove(tid)
                try:
                    self.study.finalize(trial=self._all_trials[tid].get('trial'),
                                        status=self._trial_status[status])
                    self.study.to_csv()
                except ValueError as e:
                    warn_msg = str(e)
                    warn_msg += ("\nRelevant results not found in database."
                                 " Check that Client has correct host/port, is"
                                 " submitting metrics and did not crash."
                                 " Trial script output is in: ")
                    warn_msg += os.path.join(self.study.output_dir, 'sge', 'trial_{}.out'.format(tid))
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
                self.database.add_for_stopping(tid)
                self._queued_for_stopping.add(tid)

    def submit_new_trials(self):
        """
        Get new trial and submit it to the job scheduler.
        """
        while len(self._active_trials) < self.max_concurrent:
            next_trial = self.study.get_suggestion()

            # Check if algorithm is done.
            if not next_trial:
                self._done = True
                break
            
            submit_msg = "\n" + "-"*55 + "\n" + "Submitting Trial {}:\n".format(next_trial.id)
            for pname, pval in next_trial.parameters.items():
                submit_msg += "\t{0:15}={1:>31}\n".format(str(pname), str(pval))
            submit_msg += "-"*55 + "\n"
            logger.debug(submit_msg)

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
        logger.debug("Running Loop")
        while not self._done or len(self._active_trials) != 0:
            self.update_results()

            self.update_active_trials()

            self.stop_bad_performers()

            self.submit_new_trials()

            # logger.info("Best results so far:\n"
            #             "{}".format(self.study.get_best_result()))

            # logger.info(self.study.results)
            time.sleep(5)


def optimize(parameters, algorithm, lower_is_better, filename, output_dir,
             scheduler, max_concurrent=1, db_port=None, stopping_rule=None,
             dashboard_port=None, resubmit_failed_trials=False, verbose=1):
    """
    Runs a Study with a scheduler and automatically runs a database in the
    background.

    # Arguments:
        algorithm (sherpa.algorithms.Algorithm): takes results table and returns
            parameter set.
        parameters (list[sherpa.Parameter]): parameters being optimized.
        lower_is_better (bool): whether lower objective values are better.
        filename (str): the name of the file which is called to evaluate
            configurations
        output_dir (str): where scheduler and database files will be stored.
        scheduler (sherpa.Scheduler): a scheduler.
        max_concurrent (int): the number of trials that will be evaluated in
            parallel.
        db_port (int): port to run the database on.
        stopping_rule (sherpa.stopping_rules.StoppingRule): rule for stopping
            trials prematurely.
        dashboard_port (int): port to run the dashboard web-server on.
        resubmit_failed_trials (bool): whether to resubmit a trial if it failed.
        verbose (int, default=1): whether to print submit messages (0=no, 1=yes).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if verbose == 0:
        logger.setLevel(level=logging.INFO)
        logging.getLogger('dblogger').setLevel(level=logging.WARNING)

    study = Study(parameters=parameters,
                  algorithm=algorithm,
                  lower_is_better=lower_is_better,
                  stopping_rule=stopping_rule,
                  dashboard_port=dashboard_port,
                  output_dir=output_dir)

    if not db_port:
        db_port = port_finder(27000, 28000)

    with Database(db_dir=output_dir, port=db_port) as db:
        runner = Runner(study=study,
                        scheduler=scheduler,
                        database=db,
                        max_concurrent=max_concurrent,
                        command=' '.join(['python', filename]),
                        resubmit_failed_trials=resubmit_failed_trials)
        runner.run_loop()
    return study.get_best_result()


def port_finder(start, end):
    """
    Helper function to find free port in range.
    
    # Arguments:
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
        raise('Hostname could not be resolved. Exiting')
        
    except socket.error:
        raise("Couldn't connect to server")


class Parameter(object):
    """
    Defines a hyperparameter with a name, type and associated range.
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

        # Arguments:
            config (dict): parameter config of the format
                {'name': '<name>',
                 'type': '<continuous/discrete/choice>',
                 'range': [<value1>, <value2>, ... ],
                 'scale': <'log' to sample continuous/discrete from log-scale>}

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
        super(Continuous, self).__init__(name, range)
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
        super(Discrete, self).__init__(name, range)
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
        super(Choice, self).__init__(name, range)

    def sample(self):
        i = numpy.random.randint(low=0, high=len(self.range))
        return self.range[i]
