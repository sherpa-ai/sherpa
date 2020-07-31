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
import subprocess
import re
import sys
import os
import logging


logger = logging.getLogger(__name__)


class _JobStatus(object):
    """
    Job status used internally to classify jobs into categories.
    """
    finished = 1
    running = 2
    failed = 3
    queued = 4
    killed = 5
    other = 6


class Scheduler(object):
    """
    The job scheduler gives an API to submit jobs, retrieve statuses of specific
    jobs, and kill a job.
    """
    def __init__(self):
        pass

    def submit_job(self, command, env={}, job_name=''):
        """
        Submits a job to the scheduler.

        Args:
            command (list[str]): components to the command to run by the
                scheduler e.g. ``["python", "train.py"]``
            env (dict): environment variables to pass to the job.
            job_name (str): this specifies a name for the job and its output
                directory.

        Returns:
            str: a job ID, used for getting the status or killing the job.
        """
        pass

    def get_status(self, job_id):
        """
        Obtains the current status of the job.


        Args:
            job_id (str): identifier returned when submitting the job.

        Returns:
            sherpa.schedulers._JobStatus: the job-status.
        """
        pass

    def kill_job(self, job_id):
        """
        Kills a given job.

        Args:
            job_id (str): identifier returned when submitting the job.
        """
        pass


class LocalScheduler(Scheduler):
    """
    Runs jobs locally as a subprocess.

    Args:
        submit_options (str): options appended before the command.
        resources (list[str]): list of resources that will be passed as
            SHERPA_RESOURCE environment variable. If no resource is 
            available '' will be passed.
    """
    def __init__(self, submit_options='', output_dir='', resources=None):
        self.output_dir = output_dir
        self.jobs = {}
        self.resources = resources
        self.resource_by_job = {}
        self.output_files = {}
        self.submit_options = submit_options
        self.decode_status = {0: _JobStatus.finished,
                              -15: _JobStatus.killed}
        self.output_dir = output_dir

    def submit_job(self, command, env={}, job_name=''):
        outdir = os.path.join(self.output_dir, 'jobs')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
            
        env.update(os.environ.copy())
        if self.resources is not None:
            env['SHERPA_RESOURCE'] = str(self.resources.pop())
        else:
            env['SHERPA_RESOURCE'] = ''
            
        f = open(os.path.join(outdir, '{}.out'.format(job_name)), 'w')
        optns = self.submit_options.split(' ') if self.submit_options else []
        process = subprocess.Popen(optns + command, env=env, stderr=f, stdout=f)
        self.jobs[process.pid] = process
        self.output_files[process.pid] = f
        if self.resources is not None:
            self.resource_by_job[process.pid] = env['SHERPA_RESOURCE']
        return process.pid

    def get_status(self, job_id):
        process = self.jobs.get(job_id)
        if not process:
            raise ValueError("Job not found.")
        status = process.poll()
        if status is None:
            return _JobStatus.running
        else:
            if job_id in self.resource_by_job:
                resource = self.resource_by_job.pop(job_id)
                self.resources.append(resource)
                
            if job_id in self.output_files:
                f = self.output_files.pop(process.pid)
                f.close()
            return self.decode_status.get(status, _JobStatus.other)

    def kill_job(self, job_id):
        process = self.jobs.get(job_id)
        if not process:
            raise ValueError("Job not found.")
        process.terminate()


class SGEScheduler(Scheduler):
    """
    Submits jobs to SGE, can check on their status, and kill jobs.

    Uses ``drmaa`` Python library. Due to the way SGE works it cannot
    distinguish between a failed and a completed job.

    Args:
        submit_options (str): command line options such as queue ``-q``, or
            ``-P`` for project, all written in one string.
        environment (str): the path to a file that contains environment
            variables; will be sourced before job is run.
        output_dir (str): path to directory in which ``stdout`` and ``stderr``
            will be written to. If not specified this will use the same as
            defined for the study.
    """
    def __init__(self, submit_options, environment, output_dir=''):
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
        # Create temp directory.
        outdir = os.path.join(self.output_dir, 'jobs')
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
        job_script += " ".join(command)  # 'python file.py args...'

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
            job_ids (str): SGE process ID.

        Returns:
            sherpa.schedulers._JobStatus: The job status.
        """
        with self.drmaa.Session() as s:
            try:
                status = s.jobStatus(str(job_id))
            except self.drmaa.errors.InvalidJobException:
                return _JobStatus.finished
        s = self.decode_status.get(status)
        if s == _JobStatus.finished and job_id in self.killed_jobs:
            s = _JobStatus.killed
        return s

    def kill_job(self, job_id):
        """
        Kills a job submitted to SGE.

        Args:
            job_id (str): the SGE process ID of the job.
        """
        logger.info("Killing job {}".format(job_id))
        with self.drmaa.Session() as s:
            s.control(job_id, self.drmaa.JobControlAction.TERMINATE)
        # TODO: what happens when job doesn't exist - then we don't want to add
        self.killed_jobs.add(job_id)


class SLURMScheduler(Scheduler):
    """
    Submits jobs to SLURM, can check on their status, and kill jobs.

    Uses ``drmaa`` Python library.

    Args:
        submit_options (str): command line options such as queue ``-q``,
             all written in one string.
        environment (str): the path to a file that contains environment
            variables; will be sourced before job is run.
        output_dir (str): path to directory in which ``stdout`` and ``stderr``
            will be written to. If not specified this will use the same as
            defined for the study.
    """
    def __init__(self, submit_options, environment, output_dir=''):
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
        # Create temp directory.
        logger.info('\nSUBMITTING JOB in submit_job')

        outdir = os.path.join(self.output_dir, 'jobs')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        job_name = job_name or str(self.count)
        slurmoutfile = os.path.join(outdir, '{}.out'.format(job_name))
        try:
            os.remove(slurmoutfile)
        except OSError:
            pass

        # Create bash script that sources environment and runs python script.
        job_script = '#!/bin/bash\n'
        if self.environment:
            job_script += 'source %s\n' % self.environment
        job_script += 'echo "Running from" ${HOSTNAME}\n'
        for var_name, var_value in env.items():
            job_script += 'export {}={}\n'.format(var_name, var_value)
        job_script += " ".join(command)  # 'python file.py args...'

        # Submit command to SLURM.
        # Note: submitting job using drmaa didn't work because we weren't able
        # to specify options.
        submit_command = 'sbatch --chdir={} --output={} --error={} {}'.format(
            os.getcwd(), slurmoutfile, slurmoutfile, self.submit_options)
        assert ' -cwd' not in submit_command

        # Submit using subprocess so we can get SLURM process ID.
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
            str: SLURM process ID.
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
        output_regexp = r'Submitted batch job (\d+)'
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
            job_ids (str): SLURM process ID.

        Returns:
            sherpa.schedulers._JobStatus: The job status.
        """
        with self.drmaa.Session() as s:
            try:
                status = s.jobStatus(str(job_id))
            except self.drmaa.errors.InvalidJobException:
                return _JobStatus.finished
        s = self.decode_status.get(status)
        if s == _JobStatus.finished and job_id in self.killed_jobs:
            s = _JobStatus.killed
        return s

    def kill_job(self, job_id):
        """
        Kills a job submitted to SLURM.

        Args:
            job_id (str): the SLURM process ID of the job.
        """
        logger.info("Killing job {}".format(job_id))
        with self.drmaa.Session() as s:
            s.control(job_id, self.drmaa.JobControlAction.TERMINATE)
        # TODO: what happens when job doesn't exist - then we don't want to add
        self.killed_jobs.add(job_id)
