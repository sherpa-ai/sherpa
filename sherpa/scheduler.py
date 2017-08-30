import os
import sys
import re
import subprocess
import inspect
import time
import pickle
import importlib
import abc
from collections import defaultdict
import multiprocessing as mp

class AbstractScheduler(object):
    def __init__(self):
        self.queue = mp.Queue() # Subprocess results returned here.
        self.active_processes = {}
        
    def start_subprocess(self, filename, run_id, hp, epochs, modelfile, historyfile):
        # Start subprocess to perform filename.main() with hyperparameters hp.
        # Overwrites modelfile and historyfile.
        assert type(run_id) == str
        assert type(epochs) == int
        p = mp.Process(target=self._subprocess, args=(filename, run_id, hp, epochs, modelfile, historyfile))
        p.start()
        self.active_processes[run_id] = p
        return
    
    def get_all_from_queue(self):
        # Collect any results in the queue.
        # Each result consists of the following:
        # run_id  = '%d_%d' unique to each experiment.
        # hp      = If None or empty, existing hp not overwritten in ResultsTable.
        # rval    = Return value of experiment. Not used.
        rvals = {}
        while not self.queue.empty():
            run_id, rval = self.queue.get() 
            assert type(run_id) == str, run_id
            assert run_id in self.active_processes, (run_id, self.get_active_processes())
            p = self.active_processes.pop(run_id)
            p.join()  # Process should be finished.
            rvals[run_id] = rval
        return rvals
    
    def queue_is_empty(self):
        return self.queue.empty()

    def get_active_processes(self):
        return list(self.active_processes.keys())    

    @abc.abstractmethod
    def _subprocess(self, filename, run_id, hp, epochs, modelfile, historyfile):
        '''
        Run experiment in subprocess,
        updates modelfile and historyfile,
        and puts (run_id, rval) pair in the queue when done.
        '''
        return
    
   
class LocalScheduler(AbstractScheduler):
    ''' Runs jobs as subprocesses on local machine.'''
    def __init__(self, **kwargs):
        super(LocalScheduler, self).__init__(**kwargs)
    
    def _subprocess(self, filename, run_id, hp, epochs, modelfile, historyfile):
        # Run experiment in subprocess on this machine.
        module = importlib.import_module(filename.rsplit('.', 1)[0])  # Must remove '.py' from file path.
        rval = module.main(hp=hp, epochs=epochs, modelfile=modelfile, historyfile=historyfile, verbose=2)
        #rval = None
        self.queue.put((run_id, rval))
        
class SGEScheduler(AbstractScheduler):
    ''' Submits jobs to SGE.'''
    def __init__(self, dir, environment, submit_options, **kwargs):
        self.dir   = dir
        self.environment = environment
        self.submit_options = submit_options
        super(SGEScheduler, self).__init__(**kwargs)
        
    def _subprocess(self, filename, run_id, hp, epochs, modelfile, historyfile):
        # Submit experiment to SGE and return when finished.
        # Process waits for SGE job to complete, then puts to queue.
        # However, it doesn't capture the return value.

        # Create temp directory.
        outdir = os.path.join(self.dir, 'output')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        # Create python script that can run job with specified hyperparameters.
        python_script = 'import %s as mymodule\n' % filename.rsplit('.', 1)[0]  # Module.
        python_script += 'rval=mymodule.main(modelfile=\'%s\', historyfile=\'%s\', hp=%s, epochs=%d, verbose=2)' % (
                            modelfile, historyfile, hp, epochs)
        python_script_file = os.path.join(outdir, run_id + '.py')
        with open(python_script_file, 'w') as fid:
            fid.write(python_script)

        # Create bash script that runs python script.
        sgeoutfile = os.path.join(outdir, run_id + '.out')
        try:
            os.remove(sgeoutfile)
        except:
            pass
        job_script = '#$ -S /bin/bash\n'
        if self.environment:
            job_script += 'source %s\n' % self.environment  # Set environment variables.
        job_script += 'echo "Running from" ${HOSTNAME}\n'
        job_script += 'python %s\n' % python_script_file  # How do we pass args? via python file.
        # with open(job_script_file, 'w') as fid:
        #    fid.write(job_script)

        # Submit command to SGE.
        # Note: submitting job using drmaa didn't work because we weren't able to specify options.
        # submit_options = '-N myjob -P turbomole_geopt.p -q arcus.q -l hostname=\'(arcus-1|arcus-2|arcus-3)\'' # This works.
        submit_command = 'qsub -S /bin/bash -wd {} -j y -o {} -e {} {}'.format(
                          os.getcwd(), sgeoutfile, sgeoutfile, self.submit_options)
        assert ' -cwd' not in submit_command
        process_id = self._submit_job(submit_command, job_script)  # Submit using subprocess so we can get SGE process ID.

        print('\t{}: job submitted for run_id {}'.format(process_id, run_id))

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
        if not os.path.isfile(historyfile):
            raise Exception('Job {}, run_id {} failed. (No historyfile {}.) \\
                             See SGE output in {}.'.format(
                process_id, run_id, historyfile, sgeoutfile))
        # TODO: Find a way to confirm that this subprocess succeeded.

        # Let parent process know that this job is done.
        rval = None # TODO: Figure out how to get this rval.
        self.queue.put((run_id, rval))  # See _read_queue for details.
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

    def _alive(self, process_id):
        # NOT USED.
        # This wastes a bit of time, but prevents
        # objects than inherit and don't use DRMAA from
        # having a dependency on this library
        # I am sure there is a way to get the best of both
        # worlds but this is the cleanest
        import drmaa

        s = drmaa.Session()
        s.initialize()

        try:
            status = s.jobStatus(str(process_id))
        except:
            # job not found
            sys.stderr.write("EXC: %s\n" % str(sys.exc_info()[0]))
            sys.stderr.write(
                "Could not find job for rocess id %d\n" % process_id)
            try:
                s.exit()
            except:
                pass
            return False

        if status in [drmaa.JobState.QUEUED_ACTIVE, drmaa.JobState.RUNNING]:
            alive = True

        elif status == drmaa.JobState.DONE:
            sys.stderr.write(
                "Process %d complete but not yet updated.\n" % process_id)
            alive = True

        elif status == drmaa.JobState.UNDETERMINED:
            sys.stderr.write(
                "Process %d in undetermined state.\n" % process_id)
            alive = False

        elif status in [drmaa.JobState.SYSTEM_ON_HOLD,
                        drmaa.JobState.USER_ON_HOLD,
                        drmaa.JobState.USER_SYSTEM_ON_HOLD,
                        drmaa.JobState.SYSTEM_SUSPENDED,
                        drmaa.JobState.USER_SUSPENDED]:
            sys.stderr.write("Process is held or suspended.\n" % process_id)
            alive = False

        elif status == drmaa.JobState.FAILED:
            sys.stderr.write("Process %d failed.\n" % process_id)
            alive = False

        # try to close session
        try:
            s.exit()
        except:
            pass

        return alive
