import os
import pytest
import sherpa
import pandas
import collections
try:
    import unittest.mock as mock
except ImportError:
    import mock
import logging
import tempfile
import shutil
import pymongo
import time
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.fixture
def test_trial():
    p = {'a': 1, 'b': 2}
    t = sherpa.Trial(1, p)
    assert t.id == 1
    assert t.parameters == p
    yield t


@pytest.fixture
def test_dir():
    dirpath = tempfile.mkdtemp()
    yield dirpath
    print("deleting tempdir")
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
    mock_algorithm.get_suggestion.assert_called_with(s.parameters, s.results)

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


def test_database(test_dir, test_trial):
    with sherpa.Database(test_dir) as db:
        db.enqueue_trial(test_trial)

        client = sherpa.Client(port=27017, connectTimeoutMS=2000)

        t = client.get_trial()
        assert t.id == 1
        assert t.parameters == {'a': 1, 'b': 2}

        client.send_metrics(trial=t, iteration=1,
                            objective=0.1, context={'other_metric': 0.2})

        new_results = db.get_new_results()
        logger.debug(new_results)
        assert new_results == [{'context': {'other_metric': 0.2},
                                'iteration': 1,
                                'objective': 0.1,
                                'parameters': {'a': 1, 'b': 2},
                                'trial_id': 1}]

    # with pytest.raises(pymongo.errors.ServerSelectionTimeoutError):
    #     client.get_trial()


def test_sge_scheduler(test_dir):
    if not os.environ.get("HOSTNAME") == "nimbus":
        return

    with open(os.path.join(test_dir, "sleeper.sh"), 'w') as f:
        f.write("sleep 10s\n")

    env = '/home/lhertel/profiles/main.profile'
    sge_options = '-N sherpaMNIST -P arcus.p -q arcus-ubuntu.q -l hostname=\'(arcus-5|arcus-6|arcus-8|arcus-9)\''

    s = sherpa.SGEScheduler(environment=env,
                            submit_options=sge_options)

    job_id = s.submit_job("sh {}/sleeper.sh".format(test_dir))

    assert s.get_status([job_id]) != 'failed/done'

    time.sleep(10)

    assert s.get_status([job_id]) == 'failed/done'
