import os
import sherpa
import sherpa.schedulers
import socket
import tempfile
import time
import logging
import shutil
from test_sherpa import test_dir


logging.basicConfig(level=logging.DEBUG)
testlogger = logging.getLogger(__name__)

def test_sge_scheduler():
    host = socket.gethostname()
    if (not host == "nimbus") and (not host.startswith('arcus')):
        return

    test_dir = tempfile.mkdtemp(dir=".")

    trial_script = "import time, os\n"
    trial_script += "assert os.environ.get('SHERPA_TRIAL_ID') == '3'\n"
    trial_script += "assert os.environ.get('SHERPA_HOST') == {}\n".format(host)
    trial_script += "time.sleep(5)\n"

    with open(os.path.join(test_dir, "test.py"), 'w') as f:
        f.write(trial_script)

    env = '/home/lhertel/profiles/python3env.profile'
    sge_options = '-N sherpaSchedTest -P arcus.p -q arcus.q -l hostname=\'({})\''.format(os.environ['HOSTNAME'])

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
        assert s.get_status(job_id) == sherpa.schedulers.JobStatus.finished

    finally:
        shutil.rmtree(test_dir)


def test_local_scheduler(test_dir):
    trial_script = "import time, os\n"
    trial_script += "time.sleep(3)\n"
    trial_script += "assert os.environ.get('SHERPA_TRIAL_ID') == '3'\n"

    with open(os.path.join(test_dir, "test.py"), 'w') as f:
        f.write(trial_script)

    s = sherpa.schedulers.LocalScheduler()

    job_id = s.submit_job("python {}/test.py".format(test_dir),
                          env={'SHERPA_TRIAL_ID': '3'})

    assert s.get_status(job_id) == sherpa.schedulers._JobStatus.running

    time.sleep(5)
    testlogger.debug(s.get_status(job_id))
    assert s.get_status(job_id) == sherpa.schedulers._JobStatus.finished

    job_id = s.submit_job("python {}/test.py".format(test_dir))
    time.sleep(1)
    s.kill_job(job_id)
    time.sleep(1)
    testlogger.debug(s.get_status(job_id))
    assert s.get_status(job_id) == sherpa.schedulers._JobStatus.killed

