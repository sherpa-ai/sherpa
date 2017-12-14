import subprocess
import drmaa
import re
import sys
import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Scheduler(object):
    """
    Abstract interface to a job scheduler.

    The job scheduler gives an API to submit jobs and retrieve statuses of all
    jobs.
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

    def get_status(self, process_ids):
        """
        # Returns:
            (dict) of process_id keys to respective status
        """
        pass


class SGEScheduler(Scheduler):
    """
    SGE scheduler.

    Allows to submit jobs to SGE and check on their status. Note: cannot
    distinguish between a failed and a completed job.
    """
    def __init__(self, submit_options, environment):
        self.count = 0
        self.submit_options = submit_options
        self.environment = environment

    def submit_job(self, command):
        """
        Submit experiment to SGE and return when finished.

        Process waits for SGE job to complete, then puts to queue. However,
        it doesn't capture the return value.
        """

        # Create temp directory.
        outdir = os.path.join(self.dir, 'sge')
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
        process_id = self._submit_job(submit_command, job_script)

        logger.info('\t{}: job submitted'.format(process_id))

        return process_id

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
        process.stdin.close()
        output_regexp = r'Your job (\d+)'
        # Parse out the process id from text
        match = re.search(output_regexp, output)
        if match:
            return int(match.group(1))
        else:
            sys.stderr.write(output)
            return None

    def get_status(self, process_ids):
        """
        # Arguments:
            process_ids (list[str]): list of SGE process IDs.

        # Returns:
            (list[?]) list of statuses.
        """
        statuses = {pid: None for pid in process_ids}
        with drmaa.Session() as s:
            for pid in process_ids:
                try:
                    status = s.jobStatus(str(pid))
                except drmaa.errors.InvalidJobException:
                    status = 'failed/done'
                statuses[pid] = status
        return statuses

# SGE codes
# TODO: make Sherpa enumerable with states and code into that
decodestatus = {
    drmaa.JobState.UNDETERMINED: 'process status cannot be determined',
    drmaa.JobState.QUEUED_ACTIVE: 'job is queued and active',
    drmaa.JobState.SYSTEM_ON_HOLD: 'job is queued and in system hold',
    drmaa.JobState.USER_ON_HOLD: 'job is queued and in user hold',
    drmaa.JobState.USER_SYSTEM_ON_HOLD: 'job is queued and in user and system hold',
    drmaa.JobState.RUNNING: 'job is running',
    drmaa.JobState.SYSTEM_SUSPENDED: 'job is system suspended',
    drmaa.JobState.USER_SUSPENDED: 'job is user suspended',
    drmaa.JobState.DONE: 'job finished normally',
    drmaa.JobState.FAILED: 'job finished, but failed'}