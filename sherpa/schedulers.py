class Scheduler(object):
    """
    Abstract interface to a job scheduler.

    The job scheduler gives an API to submit jobs and retrieve statuses of all
    jobs.
    """
    def __init__(self):
        pass

    def submit(self, command):
        """
        Submits a command to the scheduler.
        """

    def get_status(self):
        """
        # Returns:
            (dict) status of all jobs.
        """