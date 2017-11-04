import os
import sys
import re
import subprocess
import time
import importlib
import abc
import multiprocessing as mp

class AbstractScheduler(object):
    """
    Abstract class for scheduler.

    Attributes:
        queue (multiprocessing.Queue): Subprocess results returned here.
        active_processes (dict): maps process index to process
    """
    def __init__(self):
        self.queue = mp.Queue()    # Subprocess results returned here.
        self.active_processes = {} # 
        self.dir = None            # Directory for scheduler files; set by mainloop with set_dir.
    
    def set_dir(self, dir):
        # Sets directory for any scheduler files.
        # Called by mainloop.
        self.dir = dir

    def start_subprocess(self, filename, index, hp, modelfile, metricsfile):
        """
        Start subprocess to call filename with hyperparameters hp.

        Overwrites modelfile and metricsfile.

        Arguments:
            filename (str): name of file that contains user training code
            index (int): process index
            hp (dict): hyperparameters
            modelfile (str): path to file from which model will be loaded
            metricsfile (str): path to file that stores history
        """
        cmd = self.args_to_cmd(filename, index, hp, modelfile, metricsfile)
        p = mp.Process(target=self._subprocess, args=(cmd, index, metricsfile))
        p.start()
        self.active_processes[index] = p
        return
    
    def get_all_from_queue(self):
        """
        Collect results in the queue.

        Each result consists of the following:
            index: Unique identifying int for each experiment.
            hp: If None or empty, existing hp not overwritten in ResultsTable.
            rval: Return value of experiment. Not used.
        """
        rvals = {}
        while not self.queue.empty():
            assert len(self.get_active_processes()) > 0
            index, rval = self.queue.get() 
            assert index in self.active_processes, (index, self.get_active_processes())
            p = self.active_processes.pop(index)
            assert index not in self.active_processes
            p.join()  # Process should be finished.
            rvals[index] = rval
        return rvals
    
    def queue_is_empty(self):
        """
        Returns:
            (bool) whether queue is empty
        """
        return self.queue.empty()

    def get_active_processes(self):
        """
        Returns:
            (list) active process indices
        """
        return list(self.active_processes.keys())    

    def args_to_cmd(self, filename, index, hp, modelfile, metricsfile):
        # Command line string (or list) to run experiment from command line.
        # Unpacks hyperparameter dict into command line arguments.
        arglist = []
        arglist += ['--index', '{}'.format(index)]
        arglist += ['--metricsfile', '{}'.format(metricsfile)]
        arglist += ['--modelfile', '{}'.format(modelfile)]
        for key, val in hp.items():
            if type(val) in [int, float, bool, str]:
                arglist += ['--{}'.format(key), str(val)]
            else:
                # Convert to string and try to reconstruct in file from ast.literal_eval.
                arglist += ['--{}'.format(key), '{}'.format(str(val).replace('\'', '\"'))]
        cmd = ['python', filename] + arglist
        return cmd

    @abc.abstractmethod
    def _subprocess(self, filename, index, hp, modelfile, metricsfile):
        """
        Run experiment in subprocess,
        updates modelfile and metricsfile,
        and puts (index, rval) pair in the queue when done.
        """
        return
    
   
class LocalScheduler(AbstractScheduler):
    """
    Runs jobs as subprocesses on local machine.
    """
    def __init__(self, **kwargs):
        super(LocalScheduler, self).__init__(**kwargs)
    
    def _subprocess(self, cmd, index, metricsfile):
        """ Run experiment in subprocess."""
        try:
            subprocess.check_call(cmd) # Raises CalledProcessError if nonzero return value.
            rval = None
            self.queue.put((index, rval))
        except subprocess.CalledProcessError as e:
            print('Following bash call failed: {}'.format(cmd))
            raise e # Or should we ignore?
        return
 
class SGEScheduler(AbstractScheduler):
    """
    Submits jobs to SGE.

    # Arguments
        environment (str): Path to an environment to be used when submitting
            jobs to SGE.
        submit_options (str): Submit options for SGE in command line flags
            format.

    """
    def __init__(self, environment, submit_options, **kwargs):
        self.environment = environment
        self.submit_options = submit_options
        super(SGEScheduler, self).__init__(**kwargs)
    
    def args_to_cmd(self, filename, index, hp, modelfile, metricsfile):
        # Command line string (or list) to run experiment from command line.
        # Unpacks hyperparameter dict into command line arguments.
        # Complex hyperparameters need to have quotations around them in the 
        # Popen.communicate() method we use here.
        arglist = []
        arglist += ['--index', '{}'.format(index)]
        arglist += ['--metricsfile', '{}'.format(metricsfile)]
        arglist += ['--modelfile', '{}'.format(modelfile)]
        for key, val in hp.items():
            if type(val) in [int, float, bool, str]:
                arglist += ['--{}'.format(key), str(val)]
            else:
                # Convert to string and try to reconstruct in file from ast.literal_eval.
                arglist += ['--{}'.format(key), '\'{}\''.format(str(val).replace('\'', '\"'))] # Different from that in LocalScheduler.
        cmd = ['python', filename] + arglist
        cmd = ' '.join(cmd)
        return cmd

    def _subprocess(self, cmd, index, metricsfile):
        """
        Submit experiment to SGE and return when finished.

        Process waits for SGE job to complete, then puts to queue. However,
        it doesn't capture the return value.
        """

        # Create temp directory.
        outdir = os.path.join(self.dir, 'sge')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        # Create python script that can run job with specified hyperparameters.
        #python_script = 'import %s as mymodule\n' % filename.rsplit('.', 1)[0]  # Module.
        #python_script += 'rval=mymodule.main(modelfile=\'%s\', metricsfile=\'%s\', hp=%s, verbose=2)' % (
        #                    modelfile, metricsfile, hp)
        #python_script_file = os.path.join(outdir, '{}.py'.format(index))
        #with open(python_script_file, 'w') as fid:
        #    fid.write(python_script)

        # Create bash script that runs python script.
        sgeoutfile = os.path.join(outdir, '{}.out'.format(index))
        try:
            os.remove(sgeoutfile)
        except:
            pass
        job_script = '#$ -S /bin/bash\n'
        if self.environment:
            job_script += 'source %s\n' % self.environment  # Set environment variables.
        job_script += 'echo "Running from" ${HOSTNAME}\n'
        #job_script += 'python %s\n' % python_script_file  # How do we pass args? via python file.
        job_script += cmd  # 'python file.py args...'
        
        # Just for debugging.
        #job_script_file = os.path.join(outdir, '{}.sh'.format(index))
        #with open(job_script_file, 'w') as fid:
        #    fid.write(job_script)
        #    fid.write(str(cmd))

        # Submit command to SGE.
        # Note: submitting job using drmaa didn't work because we weren't able to specify options.
        # submit_options = '-N myjob -P turbomole_geopt.p -q arcus.q -l hostname=\'(arcus-1|arcus-2|arcus-3)\'' # This works.
        submit_command = 'qsub -S /bin/bash -wd {} -j y -o {} -e {} {}'.format(
                          os.getcwd(), sgeoutfile, sgeoutfile, self.submit_options)
        assert ' -cwd' not in submit_command

        # Submit using subprocess so we can get SGE process ID.
        process_id = self._submit_job(submit_command, job_script)

        print('\t{}: job submitted for model id {}'.format(process_id, index))

        # Wait until SGE job finishes (either finishes or fails).
        # These status messages will help solve problems where job hangs in SGE queue.
        import drmaa
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
        s = drmaa.Session()
        s.initialize()
        status = None
        while True:
            try:
                status_prev = status
                status = s.jobStatus(str(process_id))
                # Job still exists.
                if status != status_prev:
                    print('\t{}: {}'.format(process_id, decodestatus[status]))
                time.sleep(5)
            except:
                # Job no longer exists: either done or failed.
                break
        try:
            s.exit()
        except:
            pass
        # Check that history file now exists.
        if not os.path.isfile(metricsfile):
            raise Exception('Job {}, model id {} failed. (No metricsfile {}.) \
                             \nSee SGE output in {}'.format(
                             process_id, index, metricsfile, sgeoutfile))
        # TODO: Find a way to confirm that this subprocess succeeded.

        # Let parent process know that this job is done.
        rval = None
        self.queue.put((index, rval))  # See _read_queue for details.
        return

    def _submit_job(self, submit_command, run_command):
        process = subprocess.Popen(submit_command,
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   shell=True,
                                   universal_newlines=True) # Otherwise input must be bytes.
        output, std_err = process.communicate(input=run_command)
        process.stdin.close()
        output_regexp = r'Your job (\d+)'
        # Parse out the process id from text
        match = re.search(output_regexp, output)
        try:
            return int(match.group(1))
        except:
            sys.stderr.write(output)
            return None


