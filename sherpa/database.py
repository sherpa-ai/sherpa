class Database(object):
    """
    Manages a Mongo-DB for storing metrics and delivering parameters to trials.

    The Mongo-DB contains one database that serves as a queue of future trials,
    and one to store results active and finished trials.

    # Attributes:
        dbpath (str): the path where Mongo-DB stores its files.
        port (int): the port on which the Mongo-DB should run.
    """
    def start(self):
        """
        Runs the DB in a sub-process.
        """

    def get_new_results(self):
        """
        Checks database for new results.
        """

    def enqueue_trial(self):
        """
        Puts a new trial in the queue for workers to pop off.
        """


def register_with_study(port):
    """
    Registers a session with a Sherpa Study via the port of the database.

    This function is called from worker scripts only.

    # Arguments:
        port (int): port that database is running on.
    """


def get_trial():
    """
    Returns the next trial from a Sherpa Study.

    # Returns:
        (sherpa.Trial)
    """


def send_metrics(trial, iteration, objective, context):
    """
    Sends metrics for a trial to database.

    # Arguments:
        trial (sherpa.Trial): trial to send metrics for.
        iteration (int): the iteration e.g. epoch the metrics are for.
        objective (float): the objective value.
        context (dict): other metric-values.
    """