import logging
from pymongo import MongoClient
import subprocess
import time
import os
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
    def __init__(self, db_dir, port=27010):
        self.client = MongoClient(port=port)
        self.db = self.client.sherpa
        self.collected_results = []
        self.mongo_process = None
        self.dir = db_dir
        self.port = port

    def close(self):
        print('Closing MongoDB!')
        self.mongo_process.terminate()

    def start(self):
        """
        Runs the DB in a sub-process.
        """
        dblogger.debug("Starting MongoDB in {}!".format(self.dir))
        cmd = ['mongod',
               '--dbpath', self.dir,
               '--port', str(self.port),
               '--logpath', os.path.join(self.dir, "log.txt")]
        try:
            self.mongo_process = subprocess.Popen(cmd)
        except FileNotFoundError as e:
            raise FileNotFoundError(str(e) + "\nCheck that MongoDB is installed and in PATH.")
        time.sleep(1)
        self.check_db_status()

    def check_db_status(self):
        status = self.mongo_process.poll()
        if status:
            raise EnvironmentError("Database exited with code {}".format(status))

    def get_new_results(self):
        """
        Checks database for new results.
        """
        self.check_db_status()
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
        self.check_db_status()
        trial = {'trial_id': trial.id,
                 'parameters': trial.parameters}
        t_id = self.db.trials.insert_one(trial).inserted_id

    def add_for_stopping(self, trial_id):
        self.check_db_status()
        # dblogger.debug("Adding {} to DB".format({'trial_id': trial_id}))
        self.db.stop.insert_one({'trial_id': trial_id}).inserted_id

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
        host (str): the host that runs the database. Only needed if DB is not
            running on same machine.
        port (int): port that database is running on.
    """
    def __init__(self, **kwargs):
        host = os.environ.get('SHERPA_DB_HOST') or kwargs.get('hostname') or 'localhost'
        port = (os.environ.get('SHERPA_DB_PORT') or
                (kwargs.pop('port') if 'port' in kwargs else None) or
                27010)
        self.client = MongoClient(host, int(port), **kwargs)
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
        trial_id = int(os.environ.get('SHERPA_TRIAL_ID'))
        for _ in range(5):
            g = (entry for entry in self.db.trials.find({'trial_id': trial_id}))
            t = next(g)
            if t:
                break
            time.sleep(10)
        if not t:
            raise RuntimeError("No Trial Found!")
        return sherpa.Trial(id=t.get('trial_id'), parameters=t.get('parameters'))

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

        for entry in self.db.stop.find():
            print("Found entry {}".format(entry))
            if entry.get('trial_id') == trial.id:
                raise StopIteration("Trial listed for stopping.")