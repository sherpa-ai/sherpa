import logging
from pymongo import MongoClient
import subprocess
try:
    from subprocess import DEVNULL # py3k
except ImportError:
    import os
    DEVNULL = open(os.devnull, 'wb')
import sherpa


logging.basicConfig(level=logging.DEBUG)
dblogger = logging.getLogger(__name__)


class Database(object):
    """
    Manages a Mongo-DB for storing metrics and delivering parameters to trials.

    The Mongo-DB contains one database that serves as a queue of future trials,
    and one to store results active and finished trials.

    # Attributes:
        dbpath (str): the path where Mongo-DB stores its files.
        port (int): the port on which the Mongo-DB should run.
    """
    def __init__(self, dir, port=27010):
        self.client = MongoClient(port=port)
        self.db = self.client.sherpa
        self.collected_results = []
        self.mongo_process = None
        self.dir = dir
        self.port = port

    def close(self):
        print('Closing MongoDB!')
        self.mongo_process.terminate()

    def start(self):
        """
        Runs the DB in a sub-process.
        """
        dblogger.debug("Starting MongoDB!")
        cmd = ['mongod', '--dbpath', self.dir]
        if self.port:
            cmd.append('--port')
            cmd.append(str(self.port))
        self.mongo_process = subprocess.Popen(cmd,
                                              stdout=DEVNULL)

    def get_new_results(self):
        """
        Checks database for new results.
        """
        new_results = []
        for entry in self.db.results.find():
            result = entry
            result.pop('_id')
            if result not in self.collected_results:
                new_results.append(result)
                self.collected_results.append(result)
        return new_results

    def enqueue_trial(self, trial):
        """
        Puts a new trial in the queue for workers to pop off.
        """
        trial = {'id': trial.id,
                 'parameters': trial.parameters,
                 'used': False}
        t_id = self.db.trials.insert_one(trial).inserted_id

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class Client(object):
    """
    Registers a session with a Sherpa Study via the port of the database.

    This function is called from worker scripts only.

    # Arguments:
        port (int): port that database is running on.
    """
    def __init__(self, host='localhost', port=27010, **kwargs):
        self.client = MongoClient(host, port, **kwargs)
        self.db = self.client.sherpa

    def get_trial(self):
        """
        Returns the next trial from a Sherpa Study.

        # Arguments:
            client (sherpa.SherpaClient): the client obtained from registering with
                a study.

        # Returns:
            (sherpa.Trial)
        """
        t = self.db.trials.find_one_and_update(
            {"used": False},
            {'$set': {'used': True}}
        )
        if not t:
            raise RuntimeError("No Trial Found!")
        return sherpa.Trial(id=t.get('id'), parameters=t.get('parameters'))

    def send_metrics(self, trial, iteration, objective, context={}):
        """
        Sends metrics for a trial to database.

        # Arguments:
            client (sherpa.SherpaClient): client to the database.
            trial (sherpa.Trial): trial to send metrics for.
            iteration (int): the iteration e.g. epoch the metrics are for.
            objective (float): the objective value.
            context (dict): other metric-values.
        """
        result = {'parameters': trial.parameters,
                  'trial_id': trial.id,
                  'objective': objective,
                  'iteration': iteration,
                  'context': context}
        self.db.results.insert_one(result)
