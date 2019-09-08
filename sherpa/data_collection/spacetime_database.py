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
import logging
import numpy
import pymongo
from pymongo import MongoClient
import subprocess
import time
import os
import socket
import warnings

from spacetime import Node, Dataframe
from sherpa_datamodel import Trial_Results, Client as Trial_Results, Client_set
import random
import sys
import multiprocessing as mp
from frame_rate_keeper import FrameRateKeeper
from sherpa import Trial

try:
    from subprocess import DEVNULL # python 3
except ImportError:
    import os
    DEVNULL = open(os.devnull, 'wb')
import sherpa


dblogger = logging.getLogger(__name__)

def run_server(dataframe: Dataframe):
    dataframe.checkout()

    fr = FrameRateKeeper(60)

    # These 'Trial' will be assigned to any ML scripts that join the server, and ML scripts will return 'Results' after receiving 'Trail'
    all_trial_results = []
    for i in range(10):
        t = Trial(i, {1: i, 2: i + 1})
        all_trial_results.append(Trial_Results(t, "Trial_Results_{}".format(i)))
    dataframe.add_many(Trial_Results, all_trial_results)
    dataframe.commit()

    # For tracking changes in trails and results across while-loop iterations
    known_completed_trial_results = set()
    known_clients = set()

    print("Connect ML scripts to this server, \n"
          "and they will be assigned Trial_Results from the following list to start working on:")
    print("Trial_Results to complete: {}".format([trial.trial_id for trial in all_trial_results if not trial.completed]))

    while True:

        fr.tick()

        # Read in the current list of ML scripts, and compare it with known ML scripts from the last loop iteration.
        # Print if any ML scripts joined or done
        dataframe.checkout()
        clients = dataframe.read_all(Client)
        for client in clients:
            if client not in known_clients:
                known_clients.add(client)
                print("Client \"{}\" joined".format(client.name))

        clients_that_left = []
        for client in known_clients:
            if client not in clients:
                clients_that_left.append(client)
                print("Client \"{}\" left".format(client.name))
        for client in clients_that_left:
            known_clients.remove(client)

        # If any ML script clients are not done yet, assign a trial_result yet to be completed to them.
        for client in clients:
            if client.ready_for_new_trial_result:
                for trial_result in all_trial_results:
                    if trial_result.assigned_client == -1:
                        trial_result.assigned_client = client.client_id
                        client.assigned_trial_result = trial_result.trial_id
                        client.ready_for_new_trial_result = False
                        dataframe.commit()
                        dataframe.push()
                        print("Assigned trial_result \"{}\" to ML script {}".format(trial_result.name, client.name))
                        break

        # If any results are newly done, print them out
        completed_trial_results = (trial_result for trial_result in all_trial_results if trial_result.completed) #here we need to change
        trials_changed = False
        for trial_result in completed_trial_results:
            if trial_result not in known_completed_trial_results:

                print("Finished trial_result_{} with results {}".format(trial_result.trial_id, trial_result.result))
                known_completed_trial_results.add(trial_result)
                trails_changed = True
        if trials_changed:
            print("Trial_Results to complete: {}".format([trial_result.trial_id for trial_result in all_trial_results if not trial_result.completed]))

        # Exit condition
        if len(known_completed_trial_results) == len(all_trial_results):
            print("All trial_results completed.")
            break


class _Database(object):
    """
    Manages a Mongo-DB for storing metrics and delivering parameters to trials.
    The Mongo-DB contains one database that serves as a queue of future trials
    and one to store results of active and finished trials.
    Attributes:
        dbpath (str): the path where Mongo-DB stores its files.
        port (int): the port on which the Mongo-DB should run.
        reinstantiated (bool): whether an instance of the MongoDB is being loaded.
        mongodb_args (dict): keyword arguments to MongoDB
    """
    def __init__(self, port=27010):
    #self, db_dir, port=27010, reinstantiated=False,mongodb_args={}):
        self.server_app = Node(run_server, port, Types=[Trial_Results, Client_set])
        self.server_process = None
        self.port = port
        '''
        self.client = MongoClient(port=port)
        self.db = self.client.sherpa
        self.collected_results = set()
        self.mongo_process = None
        self.dir = db_dir
        self.port = port
        self.reinstantiated = reinstantiated
        self.mongodb_args = mongodb_args
        '''


    def start(self):
        """
        Runs the server in a sub-process.
        """
        '''
        args = {"--" + k: v for k, v in self.mongodb_args.items()}
        if "--dbpath" in args:
            warnings.warn("Writing MongoDB to custom path {} instead of "
                          "output dir {}".format(args["--dbpath"], self.dir),
                          UserWarning)
        else:
            args["--dbpath"] = self.dir
        if "--logpath" in args:
            warnings.warn("Writing MongoDB logs to custom path {} instead of "
                          "output dir {}".format(
                args["--logpath"], os.path.join(self.dir, "log.txt")),
                UserWarning)
        else:
            args["--logpath"] = os.path.join(self.dir, "log.txt")
        if "--port" in args:
            warnings.warn("Starting MongoDB on port {} instead of "
                          "port {}. Set port via the db_port argument in "
                          "sherpa.optimize".format(args["--port"], self.port),
                          UserWarning)
        else:
            args["--port"] = str(self.port)
        '''
        dblogger.debug("Starting Spacetime server...{}".format(self.port))

        '''
        cmd = ['mongod']
        cmd += [str(item) for keyvalue in args.items() for item in keyvalue if item is not '']

        dblogger.debug("Starting MongoDB using command:{}".format(str(cmd)))

        try:
            self.mongo_process = subprocess.Popen(cmd)
        except FileNotFoundError as e:
            raise FileNotFoundError(str(e) + "\nCheck that MongoDB is installed and in PATH.")
        time.sleep(1)
        self.check_db_status()
        if self.reinstantiated:
            self.get_new_results()
        '''

    def close(self):
        print('Closing Server!')
        self.server_process.terminate()

    def check_db_status(self): #
        """
        Checks whether database is still running.
        """
        status = self.mongo_process.poll()
        if status:
            raise EnvironmentError("Database exited with code {}".format(status))

    def get_new_results(self):
        """
        Checks database for new results.
        Returns:
            (list[dict]) where each dict is one row from the DB.
        """
        self.check_db_status()
        new_results = []
        for entry in self.db.results.find():
            result = entry
            mongo_id = result.pop('_id')
            if mongo_id not in self.collected_results:
                new_results.append(result)
                self.collected_results.add(mongo_id)
        return new_results

    def enqueue_trial(self, trial):
        """
        Puts a new trial in the queue for trial scripts to get.
        """
        self.check_db_status()
        trial = {'trial_id': trial.id,
                 'parameters': trial.parameters}
        try:
            t_id = self.db.trials.insert_one(trial).inserted_id
        except pymongo.errors.InvalidDocument:
            new_params = {}
            for k, v in trial['parameters'].items():
                if isinstance(v, numpy.int64):
                    v = int(v)

                new_params[k] = v

            trial['parameters'] = new_params
            t_id = self.db.trials.insert_one(trial).inserted_id

    def add_for_stopping(self, trial_id): #
        """
        Adds a trial for stopping.
        In the trial-script this will raise an exception causing the trial to
        stop.
        Args:
            trial_id (int): the ID of the trial to stop.
        """
        self.check_db_status()
        dblogger.debug("Adding {} to DB".format({'trial_id': trial_id}))
        self.db.stop.insert_one({'trial_id': trial_id}).inserted_id

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def _client_app(dataframe: Dataframe, client_name: str, remote):

    # Pull in the current dataframe
    dataframe.pull()
    dataframe.checkout()

    # We wish to assign our result an id number that isn't already taken
    id_nos = {client.client_id for client in dataframe.read_all(Client_set)}
    while True:
        id_no = random.randint(0, sys.maxsize)
        if id_no not in id_nos:
            break

    # Add our worker to the dataframe so the server can see that we exist
    client = Client_set(id_no, client_name, {1: 111, 2: 222})
    dataframe.add_one(Client_set, client)
    dataframe.commit()
    dataframe.push()

    fr = FrameRateKeeper(60)

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'get_new_trial_results':

                # First check if there are any trial_results available, if there aren't, 'return'
                dataframe.pull()
                dataframe.checkout()
                trial_results_list = dataframe.read_all(Trial_Results)
                if len(trial_results_list) == 0:
                    remote.send(None)
                    continue

                # Trial_Results are available, let the server know that we would like one
                client.ready_for_new_trial_result = True
                dataframe.commit()
                dataframe.push()

                # Wait for the server to give us a new trial_results, keep pulling until the server has marked
                # us as no longer ready because it assigned us a new trial_results
                while True:
                    fr.tick()
                    dataframe.pull()
                    dataframe.checkout()

                    if not client.ready_for_new_trial_result:
                        break

                # Return the new trial_results' name
                new_trial_results = None
                for t in trial_results_list:
                    if t.trial_id == data:
                        new_trial_results = t
                        client.assigned_trial_result = t.trial_id
                        break
                if new_trial_results == None:
                    remote.send(None)
                else:
                    remote.send(new_trial_results)

            elif cmd == 'submit_result':


                # Return False if we don't have a trial_results to submit results for
                if client.assigned_trial_result == -1:
                    remote.send(False)

                assigned_trial_result = dataframe.read_one(Trial_Results, oid=client.assigned_trial_result)

                # Return False if we already submitted the final results for this test
                if assigned_trial_result.completed:
                    remote.send(False)

                # Submit out results
                assigned_trial_result.completed = True
                assigned_trial_result.result = data
                client.assigned_trial_result = -1
                dataframe.commit()
                dataframe.push()

                # Return a positive result
                remote.send(True)

            elif cmd == 'currently_has_trial_result':
                remote.send(client.assigned_trial_result != -1)

            elif cmd == 'quit':
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('KeyboardInterrupt')

    # Remove client from the dataframe so the server doesn't think we still exist
    dataframe.delete_one(Client, client)
    dataframe.commit()
    dataframe.push()

    remote.close()


def _start_client_app(remote, parent_remote, server_hostname, server_port, client_name):
    parent_remote.close()

    node = Node(_client_app,
                     dataframe=(server_hostname, server_port),
                     Types=[Trial_Results, Client])

    node.start(client_name, remote)



class Client(object):
    """
    Registers a session with a Sherpa Study via creating the Client pcc_set in
    spacetime and piping a subprocess.
    This function is called from trial-scripts only.


        TBD
    Attributes:
        host (str): the host that runs the database. Passed host, host set via
            environment variable or 'localhost' in that order.
        port (int): port that database is running on. Passed port, port set via
            environment variable or 27010 in that order.



    """
    def __init__(self, host=None, port=None, test_mode=False, client_name="some_client"):
        """
        Args:
            host (str): the host that runs the database. Generally not needed since
                the scheduler passes the DB-host as an environment variable.
            port (int): port that database is running on. Generally not needed since
                the scheduler passes the DB-port as an environment variable.
            test_mode (bool): mock the client, that is, get_trial returns a trial
                that is empty, keras_send_metrics accepts calls but does not do any-
                thing, as does send_metrics. Useful for trial script debugging.
        """
        self.test_mode = test_mode
        if not self.test_mode:
            host = host or os.environ.get('SHERPA_DB_HOST') or 'localhost'
            port = port or os.environ.get('SHERPA_DB_PORT') or 27010

            self._host = host
            self._port = port
            self._client_name = client_name

            ctx = mp.get_context('fork')
            self.remote, client_remote = ctx.Pipe(duplex=True)
            self.proc = ctx.Process(target=_start_client_app, args=(client_remote, self.remote, host, port, client_name))
            self.proc.start()
            client_remote.close()

    def get_trial(self):
        """
        Returns the next trial from a Sherpa Study.
        Returns:
            sherpa.core.Trial: The trial to run.
        """
        if self.test_mode:
            return sherpa.Trial(id=1, parameters={})

        assert os.environ.get('SHERPA_TRIAL_ID'), "Environment-variable SHERPA_TRIAL_ID not found. Scheduler needs to set this variable in the environment when submitting a job"
        trial_id = int(os.environ.get('SHERPA_TRIAL_ID'))

        print("Client {} waiting for new trial_results...".format(self._client_name))
        self.remote.send(("get_new_trial_results", trial_id))
        trial_results = self.remote.recv()
        if trial_results == None:
            raise RuntimeError("No Trial Found in the spacetime frame.")
        print("Client {} got trial_results {}".format(self._client_name, trial_results.name))

        return sherpa.Trial(id=trial_results.trial_id, parameters=trial_results.parameters)

    def send_metrics(self, trial, iteration, objective, context={}):
        """
        Sends metrics for a trial to database.
        Args:
            trial (sherpa.core.Trial): trial to send metrics for.
            iteration (int): the iteration e.g. epoch the metrics are for.
            objective (float): the objective value.
            context (dict): other metric-values.
        """
        if self.test_mode:
            return
        '''
        result = {'parameters': trial.parameters,
                  'trial_id': trial.id,
                  'objective': objective,
                  'iteration': iteration,
                  'context': context}
        '''
        # Convert float32 to float64.
        # Note: Keras ReduceLROnPlateau callback requires this.
        for k,v in context.items():
            if type(v) == numpy.float32:
                context[k] = numpy.float64(v)
        results = [('parameters', list(trial.parameters.items()),
                  ('trial_id', trial.id),
                  ('objective', objective),
                  ('iteration', iteration),
                  ('context', list(context.items()))]

        self.remote.send(("submit_result", results))
        confirmation = self.remote.recv()
        if confirmation:
            print("Successfully submitted results.")
        else:
            print("Failed to submit test results.")


    def keras_send_metrics(self, trial, objective_name, context_names=[]):
        """
        Keras Callbacks to send metrics to SHERPA.
        Args:
            trial (sherpa.core.Trial): trial to send metrics for.
            objective_name (str): the name of the objective e.g. ``loss``,
                ``val_loss``, or any of the submitted metrics.
            context_names (list[str]): names of all other metrics to be
                monitored.
        """
        import keras.callbacks
        send_call = lambda epoch, logs: self.send_metrics(trial=trial,
                                                          iteration=epoch,
                                                          objective=logs[objective_name],
                                                          context={n: logs[n] for n in context_names})
        return keras.callbacks.LambdaCallback(on_epoch_end=send_call)
