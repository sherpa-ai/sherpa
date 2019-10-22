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
import os
import sherpa
import sherpa.schedulers
import tempfile
import time
import itertools
import shutil
from testing_utils import *


# Adjust for testing
SGE_QUEUE_NAME = 'arcus.q'
SGE_PROJECT_NAME = 'arcus_cpu.p'
SGE_ENV_PROFILE = '/home/lhertel/profiles/python3env.profile'


def test_sge_scheduler():
    if not os.environ.get('SGE_ROOT'):
        testlogger.info("SGE ROOT not found. Skipping SGE scheduler test.")
        return

    test_dir = tempfile.mkdtemp(dir=".")

    trial_script = "import time, os\n"
    trial_script += "assert os.environ.get('SHERPA_TRIAL_ID') == '3'\n"
    trial_script += "assert os.environ.get('SHERPA_HOST') == {}\n".format(host)
    trial_script += "time.sleep(5)\n"

    with open(os.path.join(test_dir, "test.py"), 'w') as f:
        f.write(trial_script)

    env = SGE_ENV_PROFILE
    sge_options = '-N sherpaSchedTest -P {} -q {} -l hostname=\'({})\''.format(
        SGE_PROJECT_NAME, SGE_QUEUE_NAME, os.environ['HOSTNAME'])

    s = sherpa.schedulers.SGEScheduler(environment=env,
                                       submit_options=sge_options,
                                       output_dir=test_dir)

    job_id = s.submit_job("python {}/test.py".format(test_dir),
                          env={'SHERPA_TRIAL_ID': 3, 'SHERPA_HOST': host})

    try:
        time.sleep(2)
        assert s.get_status(job_id) == sherpa.schedulers._JobStatus.running

        time.sleep(10)
        testlogger.debug(s.get_status(job_id))
        assert s.get_status(job_id) == sherpa.schedulers._JobStatus.finished

        job_id = s.submit_job("python {}/test.py".format(test_dir))
        time.sleep(1)
        s.kill_job(job_id)
        time.sleep(3)
        testlogger.debug(s.get_status(job_id))
        assert s.get_status(job_id) == sherpa.schedulers._JobStatus.finished

    finally:
        shutil.rmtree(test_dir)


def test_local_scheduler(test_dir):
    trial_script = "import time, os\n"
    trial_script += "time.sleep(3)\n"
    trial_script += "assert os.environ.get('SHERPA_TRIAL_ID') == '3'\n"

    with open(os.path.join(test_dir, "test.py"), 'w') as f:
        f.write(trial_script)

    s = sherpa.schedulers.LocalScheduler()

    job_id = s.submit_job(["python", "{}/test.py".format(test_dir)],
                          env={'SHERPA_TRIAL_ID': '3'})

    assert s.get_status(job_id) == sherpa.schedulers._JobStatus.running

    time.sleep(5)
    testlogger.debug(s.get_status(job_id))
    assert s.get_status(job_id) in [sherpa.schedulers._JobStatus.finished, sherpa.schedulers._JobStatus.other]


    job_id = s.submit_job(["python", "{}/test.py".format(test_dir)])
    time.sleep(1)
    s.kill_job(job_id)
    time.sleep(1)
    testlogger.debug(s.get_status(job_id))
    assert s.get_status(job_id) == sherpa.schedulers._JobStatus.killed
    
def test_local_scheduler_resources(test_dir):
    def generate_test_script():
        trial_script = "import time, os, sys\n"
        trial_script += "time.sleep(3)\n"
        trial_script += "assert os.environ.get('SHERPA_TRIAL_ID') == str(sys.argv[1])\n"
        trial_script += "assert os.environ.get('SHERPA_RESOURCE') == str(sys.argv[2])\n"
        return trial_script
        
    with open(os.path.join(test_dir, "test.py"), 'w') as f:
        f.write(generate_test_script())
        
    for multiple in [1, 4]:
        
        lst = range(4)
        resources = list(itertools.chain.from_iterable(itertools.repeat(x, multiple) for x in lst))
        s = sherpa.schedulers.LocalScheduler(resources=resources)

        job_ids = []
        for id, gpu in zip(range(4), reversed(range(4))):
            for j in range(multiple):
                job_ids.append(s.submit_job(["python", "{}/test.py".format(test_dir), str(id), str(gpu)],
                                      env={'SHERPA_TRIAL_ID': str(id)}))

        assert len(s.resource_by_job) == 4*multiple
        assert len(s.output_files) == 4*multiple
        for id in job_ids:
            assert s.get_status(id) == sherpa.schedulers._JobStatus.running

        assert len(s.resources) == 0

        time.sleep(5)
        for id in job_ids:
            assert s.get_status(id) in [sherpa.schedulers._JobStatus.finished, sherpa.schedulers._JobStatus.other]
        
        assert len(s.resources) == 4*multiple