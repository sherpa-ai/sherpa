"""
Job schedulers for local and remote model training
"""
from hobbit import Repository
import multiprocessing as mp
import os


class JobScheduler(object):
    """
    Base class for specific schedulers
    """
    def __init__(self, repository):
        assert isinstance(repository, Repository)
        self.repository = repository
        return

    def submit(self, **kwargs):
        """
        Calls repository.train(arg) possibly in a separate process
        Args:
            arg: ID or hparams

        Returns:

        """
        self.repository.train(**kwargs)


class RemoteJobScheduler(JobScheduler):
    """
    Schedules jobs on remote machines
    """