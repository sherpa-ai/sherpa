from __future__ import absolute_import
from __future__ import division
from .resultstable import ResultsTable
import time
import importlib
import pickle as pkl
import os
import sys
import re
import subprocess
import inspect
from collections import defaultdict
from multiprocessing import Process, Queue


class MainLoop():
    """
    Main Loop:
    1) Query Algorithm
    2) Start Experiment (possibly asynchronously)
    3) Write Results into the ResultsTable
    """

    def __init__(self, fname, algorithm, dir='./', loss='loss',
                 results_table=None, environment=None, submit_options=''):
        assert isinstance(dir, str)
        self.fname = fname  # Module with method main(run_id, hparams) (e.g. nn.py).
        self.algorithm = algorithm  # Instantiated Sherpa Algorithm object.
        self.loss = loss  # Key in Keras history to be minimized, e.g. 'loss' or 'val_loss'.
        self.dir = dir  # Directory in which all files are stored: models, history, and other output.
        # Arguments for parallel jobs.
        self.environment = environment  # Bash script that sets environment variables for parallel jobs.
        self.submit_options = submit_options  # Command line options for submission to queueing systems for parallel jobs.
        self.job_management = 'sge'  # Used for parallel jobs. Options: 'sge' or 'local'
        # Make dir if neccessary.
        try:
            os.makedirs(self.dir)  # os.makedirs(os.path.dirname(self.dir))
        except:
            print('\nWARNING: Found existing directory %s, algorithm may make '
                  'unintended use of old results!' % self.dir)
            pass

        self.results_table = results_table if results_table is not None else ResultsTable(
            self.dir, recreate=False)
        return

    def run(self, max_concurrent=1):
        # Run main loop.
        if max_concurrent > 1:
            self.run_parallel(max_concurrent=max_concurrent)
        # Sequential loop.
        module = importlib.import_module(
            self.fname.rsplit('.', 1)[0])  # Must remove '.py' from file path.
        while True:
            # Query Algorithm
            rval = self.algorithm.next(self.results_table, pending={})
            if rval == 'stop':
                break  # Done
            elif rval == 'wait':
                raise Exception('Should not have to wait in sequential mode.')
            else:
                assert type(rval) == tuple and len(rval) == 3
                run_id, hparams, epochs = rval
                modelfile = '%s/%s_model.h5' % (self.dir, run_id)
                historyfile = '%s/%s_history.pkl' % (self.dir, run_id)
                history, initial_epoch = get_hist(hparams=hparams,
                                                  historyfile=historyfile)
                partialh = module.main(modelfile=modelfile,
                                   hparams=hparams,
                                   epochs=epochs,
                                   initial_epoch=initial_epoch,
                                   verbose=1)
                # Update ResultsTable.
                store_hist(partialh=partialh,
                           history=history,
                           historyfile=historyfile)
                with open(historyfile, 'rb') as f:
                    history = pkl.load(f)
                lowest_loss = min(history[self.loss])
                epochs_seen = len(history[self.loss])
                self.results_table.set(run_id=run_id, hparams=hparams,
                                       loss=lowest_loss, epochs=epochs_seen)

    def run_parallel(self, max_concurrent=1):
        # Use multiprocessing to run jobs in subprocesses.
        processes = {}  # Keep track of active subprocesses.
        queue = Queue()  # Subprocess results returned here.
        while True:
            # Collect any results in the queue and write directly to ResultsTable.
            self._read_queue(queue, processes)

            # Limit number of concurrent subprocesses.
            if len(processes) >= max_concurrent:
                time.sleep(5)
                continue

            # Query Algorithm about next experiment.
            rval = self.algorithm.next(self.results_table,
                                       pending=processes.keys())
            if rval == 'stop':
                # Finished.
                break
            elif rval == 'wait':
                # Wait for more jobs to complete before submitting more.
                time.sleep(5)
                continue
            else:
                # Start new experiment specified by Algorithm.
                run_id, hparams, epochs = rval
                p = self._start_subprocess(queue, rval)
                processes[run_id] = p
                time.sleep(3)  # Delay might avoid errors in gpu locking.
                assert len(processes) <= max_concurrent
        assert len(processes) == 0, processes
        assert queue.empty()

    def _read_queue(self, queue, processes):
        # Collect any results in the queue.
        # Each result consists of the following:
        # run_id  = '%d_%d' unique to each experiment.
        # hparams = If None or empty, existing hparams not overwritten in ResultsTable.
        # rval    = Return value of experiment. Not used.
        # historyfile = File where loss information is saved.
        while not queue.empty():
            run_id, hparams, rval, historyfile = queue.get()
            p = processes.pop(run_id)
            p.join()  # Process should be finished.
            # Update results table by reading historyfile.
            history = pkl.load(open(historyfile, 'r'))
            lowest_loss = min(history[self.loss])
            epochs_seen = len(history[self.loss])
            self.results_table.set(run_id=run_id, hparams=hparams,
                                   loss=lowest_loss, epochs=epochs_seen)
        return

    def _start_subprocess(self, queue, rval):
        # Start subprocess to perform experiment specified by Algorithm.
        # Overwrites modelfile and historyfile.
        assert type(rval) == tuple and len(rval) == 3
        run_id, hparams, epochs = rval
        modelfile = '%s/%s_model.h5' % (self.dir, run_id)
        historyfile = '%s/%s_history.pkl' % (self.dir, run_id)
        if self.job_management == 'local':
            p = Process(target=self._subprocess_local, args=(
            queue, run_id, modelfile, historyfile, hparams, epochs))
        elif self.job_management == 'sge':
            p = Process(target=self._subprocess_sge, args=(
            queue, run_id, modelfile, historyfile, hparams, epochs))
        else:
            raise Exception(
                'Bad value of job_management: %s' % self.job_management)
        p.start()
        return p

    def _subprocess_local(self, queue, run_id, modelfile, historyfile, hparams,
                          epochs):
        # Run experiment in subprocess on this machine.
        rval = self.module.main(modelfile=modelfile, historyfile=historyfile,
                                hparams=hparams, epochs=epochs, verbose=2)
        queue.put([run_id, hparams, rval,
                   historyfile])  # See _read_queue for details.

    def _subprocess_sge(self, queue, run_id, modelfile, historyfile, hparams,
                        epochs):
        # Submit experiment to SGE and return when finished.
        # Process waits for SGE job to complete, then puts to queue to signal that it is done.

        # Create temp directory.
        outdir = os.path.join(self.dir, 'output')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        # Create python script that can run job with specified hyperparameters.
        python_script = 'import %s as mymodule\n' % self.fname.rsplit('.', 1)[
            0]  # Module.



        python_script += inspect.getsource(get_hist)
        python_script += 'history, initial_epoch = get_hist(hparams=%s, ' \
                         'historyfile=historyfile)' \
                         ')\n' % hparams
        python_script += 'partialh=mymodule.main(\'%s\', initial_epoch=initial_epoch, hparams=%s, epochs=%d, verbose=2)' % (
        modelfile, hparams, epochs)

        python_script += inspect.getsource(store_hist)
        python_script += 'store_hist(partialh, history, historyfile)\n'

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
        submit_command = 'qsub -S /bin/bash -wd %s -j y -o %s -e %s %s' % (
        os.getcwd(), sgeoutfile, sgeoutfile, self.submit_options)
        assert ' -cwd' not in submit_command
        process_id = self._submit_job(submit_command,
                                      job_script)  # Submit using subprocess so we can get SGE process ID.
        print('\t%d: job submitted for run_id %s' % (process_id, run_id))

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
                    print('\t%d: %s' % (process_id, decodestatus[status]))
                time.sleep(5)
            except:
                # Job no longer exists: either done or failed.
                break
        try:
            s.exit()
        except:
            pass
        # Check that history file now exists.
        assert os.path.isfile(
            historyfile), 'Job %s for run_id %s must have failed. No historyfile %s' % (
        process_id, run_id, historyfile)
        # TODO: Find a way to confirm that this subprocess succeeded.

        # Let parent process know that this job is done.
        queue.put([run_id, hparams, None,
                   historyfile])  # See _read_queue for details.
        return

    def _submit_job(self, submit_command, run_command):
        process = subprocess.Popen(submit_command,
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   shell=True)
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


def get_hist(hparams, historyfile):
    if hparams is None or len(hparams) == 0:
        # Restart from modelfile and historyfile.
        with open(historyfile, 'rb') as f:
            history = pkl.load(f)
        initial_epoch = len(history['loss'])
    else:
        # Create new model.
        history = defaultdict(list)
        initial_epoch = 0
    return history, initial_epoch


def store_hist(partialh, history, historyfile):
    # partialh = partialh.history
    for k in partialh:
        history[k].extend(partialh[k])
    assert 'loss' in history, 'Sherpa requires a loss to be defined in history.'

    with open(historyfile, 'wb') as fid:
        pkl.dump(history, fid)