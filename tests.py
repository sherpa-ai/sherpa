import pytest
import sherpa
import pandas
import collections
import unittest.mock as mock
import logging
import tempfile
import shutil
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
    db = sherpa.Database(test_dir)

    db.start()

    db.enqueue_trial(test_trial)

    client = sherpa.Client(port=27017)

    t = sherpa.get_trial(client)
    assert t.id == 1
    assert t.parameters == {'a': 1, 'b': 2}

    sherpa.send_metrics(client=client, trial=t, iteration=1,
                        objective=0.1, context={'other_metric': 0.2})

    new_results = db.get_new_results()
    logger.debug(new_results)
    # assert new_results == [{}]

    del db

    client.get_trial()



if __name__ == '__main__':
    # test_trial()
    # test_parameters()
    # test_study()
    test_database(test_dir(), test_trial())