import subprocess
import re
import sys
import os
from enum import Enum
import logging


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class _JobStatus(Enum):
    finished = 1
    running = 2
    failed = 3
    queued = 4
    killed = 5
    other = 6


class Scheduler(object):
    """
    The job scheduler gives an API to submit jobs, retrieve statuses of all
    jobs and kill a job.
    """
    def __init__(self):
        pass

    def submit_job(self, command, env={}, job_name=''):
        """
        Submits a command to the scheduler.

        Args:
            command (str): the command to run by the scheduler.
        """
        pass

    def get_status(self, job_id):
        """
        Returns:
            dict: of job_id keys to respective status
        """
        pass


class LocalScheduler(Scheduler):
    """
    Runs jobs locally as a subprocess.
    """
    def __init__(self, submit_options=''):
        self.jobs = {}
        self.submit_options = submit_options
        self.decode_status = {0: _JobStatus.finished,
                              -15: _JobStatus.killed}

    def submit_job(self, command, env={}, job_name=''):
        env.update(os.environ.copy())
        optns = self.submit_options.split(' ') if self.submit_options else []
        process = subprocess.Popen(optns + command.split(' '), env=env)
        self.jobs[process.pid] = process
        return process.pid

    def get_status(self, job_id):
        process = self.jobs.get(job_id)
        if not process:
            raise ValueError("Job not found.")
        status = process.poll()
        if status is None:
            return _JobStatus.running
        else:
            return self.decode_status.get(status, _JobStatus.other)


class SGEScheduler(Scheduler):
    """
    SGE scheduler.

    Allows to submit jobs to SGE and check on their status. Note: cannot
    distinguish between a failed and a completed job.
    """
    def __init__(self, submit_options, environment, output_dir):
        self.count = 0
        self.submit_options = submit_options
        self.environment = environment
        self.output_dir = output_dir
        self.killed_jobs = set()
        self.drmaa = __import__('drmaa')
        self.decode_status = {
            self.drmaa.JobState.UNDETERMINED: _JobStatus.other,
            self.drmaa.JobState.QUEUED_ACTIVE: _JobStatus.queued,
            self.drmaa.JobState.SYSTEM_ON_HOLD: _JobStatus.other,
            self.drmaa.JobState.USER_ON_HOLD: _JobStatus.other,
            self.drmaa.JobState.USER_SYSTEM_ON_HOLD: _JobStatus.other,
            self.drmaa.JobState.RUNNING: _JobStatus.running,
            self.drmaa.JobState.SYSTEM_SUSPENDED: _JobStatus.other,
            self.drmaa.JobState.USER_SUSPENDED: _JobStatus.other,
            self.drmaa.JobState.DONE: _JobStatus.finished,
            self.drmaa.JobState.FAILED: _JobStatus.failed}

    def submit_job(self, command, env={}, job_name=''):
        """
        Submit experiment to SGE.
        """
        # Create temp directory.
        outdir = os.path.join(self.output_dir, 'sge')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        job_name = job_name or str(self.count)
        sgeoutfile = os.path.join(outdir, '{}.out'.format(job_name))
        try:
            os.remove(sgeoutfile)
        except OSError:
            pass

        # Create bash script that sources environment and runs python script.
        job_script = '#$ -S /bin/bash\n'
        if self.environment:
            job_script += 'source %s\n' % self.environment
        job_script += 'echo "Running from" ${HOSTNAME}\n'
        for var_name, var_value in env.items():
            job_script += 'export {}={}\n'.format(var_name, var_value)
        job_script += command  # 'python file.py args...'

        # Submit command to SGE.
        # Note: submitting job using drmaa didn't work because we weren't able
        # to specify options.
        submit_command = 'qsub -S /bin/bash -wd {} -j y -o {} -e {} {}'.format(
            os.getcwd(), sgeoutfile, sgeoutfile, self.submit_options)
        assert ' -cwd' not in submit_command

        # Submit using subprocess so we can get SGE process ID.
        job_id = self._submit_job(submit_command, job_script)

        logger.info('\t{}: job submitted'.format(job_id))
        self.count += 1

        return job_id

    @staticmethod
    def _submit_job(submit_command, run_command):
        """
        Args:
            submit_command (str): e.g. "qsub -N myProject ..."
            run_command (str): e.g. "python nn.py"

        Returns:
            str: SGE process ID.
        """
        process = subprocess.Popen(submit_command,
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   shell=True,
                                   universal_newlines=True)
        output, std_err = process.communicate(input=run_command)
        # output, std_err = process.communicate()
        process.stdin.close()
        output_regexp = r'Your job (\d+)'
        # Parse out the process id from text
        match = re.search(output_regexp, output)
        if match:
            return match.group(1)
        else:
            sys.stderr.write(output)
            return None

    def get_status(self, job_id):
        """
        Args:
            job_ids (list[str]): list of SGE process IDs.

        Returns:
            sherpa._JobStatus: The job status.
        """
        with self.drmaa.Session() as s:
            try:
                status = s._JobStatus(str(job_id))
            except self.drmaa.errors.InvalidJobException:
                return _JobStatus.finished
        s = self.decode_status.get(status)
        if s == _JobStatus.finished and job_id in self.killed_jobs:
            s = _JobStatus.killed
        return s

