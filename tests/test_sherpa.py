import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import sherpa
import sherpa.schedulers
import pandas
import collections
import pymongo
try:
    import unittest.mock as mock
except ImportError:
    import mock
import logging
import tempfile
import shutil
import time

logging.basicConfig(level=logging.DEBUG)
testlogger = logging.getLogger(__name__)


def test_trial():
    p = {'a': 1, 'b': 2}
    t = sherpa.Trial(1, p)
    assert t.id == 1
    assert t.parameters == p


@pytest.fixture
def test_dir():
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


def get_test_parameters():
    c = sherpa.Parameter.from_dict({'type': 'continuous',
                                    'name': 'a',
                                    'range': [1, 2]})
    cl = sherpa.Parameter.from_dict({'type': 'continuous',
                                     'name': 'b',
                                     'range': [1, 2],
                                     'scale': 'log'})
    d = sherpa.Parameter.from_dict({'type': 'discrete',
                                    'name': 'c',
                                    'range': [1, 10]})
    dl = sherpa.Parameter.from_dict({'type': 'discrete',
                                     'name': 'd',
                                     'range': [1, 10],
                                     'scale': 'log'})
    ch = sherpa.Parameter.from_dict({'type': 'choice',
                                     'name': 'e',
                                     'range': [1, 10]})
    return c, cl, d, dl, ch


def test_parameters():
    c, cl, d, dl, ch = get_test_parameters()

    assert c.name == 'a'
    assert all(1 <= c.sample() <= 2 for _ in range(10))
    assert all(1 <= cl.sample() <= 2 for _ in range(10))
    assert all(1 <= d.sample() <= 10 for _ in range(10))
    assert all(1 <= dl.sample() <= 10 for _ in range(10))
    assert all(ch.sample() in [1, 10] for _ in range(10))


def test_study():
    mock_algorithm = mock.MagicMock()
    mock_algorithm.get_suggestion.return_value = {'a': 1, 'b': 2}
    mock_stopping_rule = mock.MagicMock()

    s = sherpa.Study(parameters=get_test_parameters(),
                     algorithm=mock_algorithm,
                     stopping_rule=mock_stopping_rule,
                     lower_is_better=True)

    t = s.get_suggestion()
    assert t.id == 1
    assert t.parameters == {'a': 1, 'b': 2}
    mock_algorithm.get_suggestion.assert_called_with(s.parameters, s.results,
                                                     s.lower_is_better)

    s.add_observation(trial=t, iteration=1, objective=0.1,
                      context={'other_metrics': 0.2})
    s.add_observation(trial=t, iteration=2, objective=0.01,
                      context={'other_metrics': 0.02})
    s.finalize(trial=t, status='COMPLETED')

    expected_df = pandas.DataFrame(collections.OrderedDict(
        [('Trial-ID', [1, 1, 1]),
         ('Status', ['INTERMEDIATE', 'INTERMEDIATE', 'COMPLETED']),
         ('Iteration', [1, 2, 2]),
         ('a', [1, 1, 1]),
         ('b', [2, 2, 2]),
         ('Objective', [0.1, 0.01, 0.01]),
         ('other_metrics', [0.2, 0.02, 0.02])]
    ))

    assert s.results.equals(expected_df)


def test_database(test_dir):
    test_trial = get_test_trial()
    testlogger.debug(test_dir)
    db_port = 27011
    with sherpa.Database(test_dir, port=db_port) as db:
        testlogger.debug("Enqueuing...")
        db.enqueue_trial(test_trial)

        client = sherpa.Client(port=db_port,
                               connectTimeoutMS=100,
                               serverSelectionTimeoutMS=1000)

        testlogger.debug("Getting Trial...")
        t = client.get_trial()
        assert t.id == 1
        assert t.parameters == {'a': 1, 'b': 2}

        testlogger.debug("Sending Metrics...")
        client.send_metrics(trial=t, iteration=1,
                            objective=0.1, context={'other_metric': 0.2})

        new_results = db.get_new_results()
        testlogger.debug(new_results)
        assert new_results == [{'context': {'other_metric': 0.2},
                                'iteration': 1,
                                'objective': 0.1,
                                'parameters': {'a': 1, 'b': 2},
                                'trial_id': 1}]

        db.add_for_stopping(t.id)
        with pytest.raises(StopIteration):
            testlogger.debug("Stopping Trial...")
            client.send_metrics(trial=t, iteration=2,
                                objective=0.1, context={'other_metric': 0.2})

        # test that Sherpa raises correct error if MongoDB exits
        db2 = sherpa.Database(test_dir, port=db_port)
        with pytest.raises(OSError):
            db2.start()

    # with pytest.raises(RuntimeError):
    # with pytest.raises(pymongo.errors.ServerSelectionTimeoutError):
    #     client.get_trial()


def test_sge_scheduler():
    if not os.environ.get("HOSTNAME") == "nimbus":
        return

    test_dir = tempfile.mkdtemp(dir=".")

    with open(os.path.join(test_dir, "test.py"), 'w') as f:
        f.write("import time\ntime.sleep(5)")

    env = '/home/lhertel/profiles/main.profile'
    sge_options = '-N sherpaSchedTest -P arcus.p -q arcus.q -l hostname=\'(arcus-1|arcus-2|arcus-8|arcus-9)\''

    s = sherpa.schedulers.SGEScheduler(environment=env,
                                       submit_options=sge_options,
                                       output_dir=test_dir)

    job_id = s.submit_job("python {}/test.py".format(test_dir))

    try:
        time.sleep(2)
        assert s.get_status(job_id) == sherpa.schedulers.JobStatus.running

        time.sleep(10)
        testlogger.debug(s.get_status(job_id))
        assert s.get_status(job_id) == sherpa.schedulers.JobStatus.finished

    finally:
        shutil.rmtree(test_dir)


def test_local_scheduler(test_dir):

    with open(os.path.join(test_dir, "test.py"), 'w') as f:
        f.write("import time\ntime.sleep(3)")

    s = sherpa.schedulers.LocalScheduler()

    job_id = s.submit_job("python {}/test.py".format(test_dir))

    assert s.get_status(job_id) == sherpa.schedulers.JobStatus.running

    time.sleep(5)
    testlogger.debug(s.get_status(job_id))
    assert s.get_status(job_id) == sherpa.schedulers.JobStatus.finished


def get_test_study():
    mock_algorithm = mock.MagicMock()
    mock_algorithm.get_suggestion.return_value = {'a': 1, 'b': 2}
    mock_stopping_rule = mock.MagicMock()

    s = sherpa.Study(parameters=get_test_parameters(),
                     algorithm=mock_algorithm,
                     stopping_rule=mock_stopping_rule,
                     lower_is_better=True)

    return s


def get_test_trial(id=1):
    p = {'a': 1, 'b': 2}
    t = sherpa.Trial(id, p)
    return t


def test_runner_update_results():
    mock_db = mock.MagicMock()
    mock_db.get_new_results.return_value = [{'context': {'other_metric': 0.2},
                                         'iteration': 1,
                                         'objective': 0.1,
                                         'parameters': {'a': 1, 'b': 2},
                                         'trial_id': 1}]

    r = sherpa.Runner(study=get_test_study(), scheduler=mock.MagicMock(),
                      database=mock_db, max_concurrent=1,
                      command="python test.py")

    # new trial
    t = get_test_trial()
    r.all_trials[t.id] = {'trial': t, 'job_id': None}
    r.update_results()
    testlogger.debug(r.study.results)
    assert r.study.results['Trial-ID'].isin([1]).sum()

    # new observation
    mock_db.get_results.return_value = [{'context': {'other_metric': 0.3},
                                         'iteration': 2,
                                         'objective': 0.2,
                                         'parameters': {'a': 1, 'b': 2},
                                         'trial_id': 1}]
    r.update_results()
    assert 0.2 in r.study.results['Objective']


def test_runner_update_active_trials():
    mock_scheduler = mock.MagicMock()

    mock_study = mock.MagicMock()

    r = sherpa.Runner(study=mock_study, scheduler=mock_scheduler,
                      database=mock.MagicMock(), max_concurrent=1,
                      command="python test.py")

    t = get_test_trial()
    r.all_trials[t.id] = {'trial': t, 'job_id': None}
    r.active_trials.append(t.id)

    mock_scheduler.get_status.return_value = sherpa.schedulers.JobStatus.running
    r.update_active_trials()

    mock_scheduler.get_status.return_value = sherpa.schedulers.JobStatus.finished
    r.update_active_trials()

    mock_study.finalize.assert_called_with(trial=t, status='COMPLETED')

    assert r.active_trials == []


def test_runner_stop_bad_performers():
    r = sherpa.Runner(study=mock.MagicMock(),
                      scheduler=mock.MagicMock(),
                      database=mock.MagicMock(),
                      max_concurrent=1,
                      command="python test.py")

    # setup
    t = get_test_trial()
    r.active_trials.append(t.id)
    r.all_trials[t.id] = {'trial': t, 'job_id': '111'}

    # test that trial is stopped
    r.update_active_trials = mock.MagicMock()
    r.study.should_trial_stop.return_value = True
    r.stop_bad_performers()
    r.scheduler.kill_job.assert_called_with('111')

    # test that trial is not stopped
    r.study.should_trial_stop.return_value = False
    r.stop_bad_performers()

    # make sure trial is only killed in one case
    r.update_active_trials.assert_called_once()


def test_runner_submit_new_trials():
    mock_scheduler = mock.MagicMock()
    mock_scheduler.submit.side_effect = ['job1', 'job2', 'job3']
    mock_study = mock.MagicMock()
    mock_study.get_suggestion.side_effect = [get_test_trial(1),
                                             get_test_trial(2),
                                             get_test_trial(3)]

    r = sherpa.Runner(study=mock_study,
                      scheduler=mock_scheduler,
                      database=mock.MagicMock(),
                      max_concurrent=3,
                      command="python test.py")

    r.submit_new_trials()

    mock_scheduler.submit.has_calls([mock.call("python test.py"),
                                     mock.call("python test.py"),
                                     mock.call("python test.py")])
    assert len(r.active_trials) == 3
    assert len(r.all_trials) == 3