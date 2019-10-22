import sherpa
import sherpa.core
import sherpa.schedulers
import sherpa.database
try:
    import unittest.mock as mock
except ImportError:
    import mock
from testing_utils import *


def test_runner_update_results():
    mock_db = mock.MagicMock()
    mock_db.get_new_results.return_value = [{'context': {'other_metric': 0.2},
                                         'iteration': 1,
                                         'objective': 0.1,
                                         'parameters': {'a': 1, 'b': 2},
                                         'trial_id': 1}]

    r = sherpa.core._Runner(study=get_mock_study(), scheduler=mock.MagicMock(),
                      database=mock_db, max_concurrent=1,
                      command=["python", "test.py"])

    # new trial
    t = get_test_trial()
    r._all_trials[t.id] = {'trial': t, 'job_id': None}
    r.update_results()
    testlogger.debug(r.study.results)
    assert r.study.results['Trial-ID'].isin([1]).sum()

    # new observation
    mock_db.get_new_results.return_value = [{'context': {'other_metric': 0.3},
                                         'iteration': 2,
                                         'objective': 0.2,
                                         'parameters': {'a': 1, 'b': 2},
                                         'trial_id': 1}]
    r.update_results()
    assert 0.2 in list(r.study.results['Objective'])


def test_runner_update_active_trials():
    mock_scheduler = mock.MagicMock()

    mock_study = mock.MagicMock()

    r = sherpa.core._Runner(study=mock_study, scheduler=mock_scheduler,
                      database=mock.MagicMock(), max_concurrent=1,
                      command=["python", "test.py"])

    t = get_test_trial()
    r._all_trials[t.id] = {'trial': t, 'job_id': None}
    r._active_trials.append(t.id)

    mock_scheduler.get_status.return_value = sherpa.schedulers._JobStatus.running
    r.update_active_trials()

    mock_scheduler.get_status.return_value = sherpa.schedulers._JobStatus.finished
    r.update_active_trials()

    mock_study.finalize.assert_called_with(trial=t, status='COMPLETED')

    assert r._active_trials == []


def test_runner_stop_bad_performers():
    r = sherpa.core._Runner(study=mock.MagicMock(),
                      scheduler=mock.MagicMock(),
                      database=mock.MagicMock(),
                      max_concurrent=1,
                      command=["python", "test.py"])

    # setup
    t = get_test_trial()
    r._active_trials.append(t.id)
    r._all_trials[t.id] = {'trial': t, 'job_id': '111'}

    # test that trial is stopped
    r.update_active_trials = mock.MagicMock()
    r.study.should_trial_stop.return_value = True
    r.stop_bad_performers()
    r.scheduler.kill_job.assert_called_with('111')

    # test that trial is not stopped
    r.study.should_trial_stop.return_value = False
    r.stop_bad_performers()


def test_runner_submit_new_trials():
    mock_scheduler = mock.MagicMock()
    mock_scheduler.submit_job.side_effect = ['job1', 'job2', 'job3']
    mock_study = mock.MagicMock()
    mock_study.get_suggestion.side_effect = [get_test_trial(1),
                                             get_test_trial(2),
                                             get_test_trial(3)]

    r = sherpa.core._Runner(study=mock_study,
                      scheduler=mock_scheduler,
                      database=mock.MagicMock(),
                      max_concurrent=3,
                      command=["python", "test.py"])

    r.submit_new_trials()

    mock_scheduler.submit_job.has_calls([mock.call(["python", "test.py"]),
                                         mock.call(["python", "test.py"]),
                                         mock.call(["python", "test.py"])])
    assert len(r._active_trials) == 3
    assert len(r._all_trials) == 3