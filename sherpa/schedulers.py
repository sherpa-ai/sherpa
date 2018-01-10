import subprocess
import re
import sys
import os
from enum import Enum
import logging


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class JobStatus(Enum):
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

    def submit_job(self, command):
        """
        Submits a command to the scheduler.

        # Arguments:
            command (str): the command to run by the scheduler.
        """
        pass

    def get_status(self, job_id):
        """
        # Returns:
            (dict) of job_id keys to respective status
        """
        pass

    # def kill_job(self, job_id):
    #     """
    #     Kills a given jobs.
    #
    #     # Arguments:
    #         job_id (str): the id of the job to be killed.
    #     """
    #     pass


class LocalScheduler(Scheduler):
    """
    Runs jobs locally as a subprocess.
    """
    def __init__(self):
        self.jobs = {}
        self.decode_status = {0: JobStatus.finished,
                              -15: JobStatus.killed}

    def submit_job(self, command):
        process = subprocess.Popen(command.split(' '))
        self.jobs[process.pid] = process
        return process.pid

    def get_status(self, job_id):
        process = self.jobs.get(job_id)
        if not process:
            raise ValueError("Job not found.")
        status = process.poll()
        if status is None:
            return JobStatus.running
        else:
            return self.decode_status.get(status, JobStatus.other)

    # def kill_job(self, job_id):
    #     process = self.jobs.get(job_id)
    #     if not process:
    #         raise ValueError("Job not found.")
    #     process.terminate()


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
            self.drmaa.JobState.UNDETERMINED: JobStatus.other,
            self.drmaa.JobState.QUEUED_ACTIVE: JobStatus.queued,
            self.drmaa.JobState.SYSTEM_ON_HOLD: JobStatus.other,
            self.drmaa.JobState.USER_ON_HOLD: JobStatus.other,
            self.drmaa.JobState.USER_SYSTEM_ON_HOLD: JobStatus.other,
            self.drmaa.JobState.RUNNING: JobStatus.running,
            self.drmaa.JobState.SYSTEM_SUSPENDED: JobStatus.other,
            self.drmaa.JobState.USER_SUSPENDED: JobStatus.other,
            self.drmaa.JobState.DONE: JobStatus.finished,
            self.drmaa.JobState.FAILED: JobStatus.failed}

    def submit_job(self, command):
        """
        Submit experiment to SGE.
        """
        # Create temp directory.
        outdir = os.path.join(self.output_dir, 'sge')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        sgeoutfile = os.path.join(outdir, '{}.out'.format(self.count))
        try:
            os.remove(sgeoutfile)
        except OSError:
            pass

        # Create bash script that sources environment and runs python script.
        job_script = '#$ -S /bin/bash\n'
        if self.environment:
            job_script += 'source %s\n' % self.environment
        job_script += 'echo "Running from" ${HOSTNAME}\n'
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
        # Arguments:
            submit_command (str): e.g. "qsub -N myProject ..."
            run_command (str): e.g. "python nn.py"

        # Returns:
            (str) SGE process ID.
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
        # Arguments:
            job_ids (list[str]): list of SGE process IDs.

        # Returns:
            JobStatus
        """
        with self.drmaa.Session() as s:
            try:
                status = s.jobStatus(str(job_id))
            except self.drmaa.errors.InvalidJobException:
                return JobStatus.finished
        s = self.decode_status.get(status)
        if s == JobStatus.finished and job_id in self.killed_jobs:
            s = JobStatus.killed
        return s

    # def kill_job(self, job_id):
    #     """
    #     Kills a job submitted to SGE.
    #
    #     # Arguments:
    #         job_id (str): the SGE ID of the job.
    #     """
    #     logger.info("Killing job {}".format(job_id))
    #     with self.drmaa.Session() as s:
    #         s.control(job_id, self.drmaa.JobControlAction.TERMINATE)
    #     # TODO: what happens when job doesn't exist - then we don't want to add
    #     self.killed_jobs.add(job_id)

