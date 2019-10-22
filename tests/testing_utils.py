import pytest
import sherpa
import sherpa.core
import sherpa.schedulers
import sherpa.database
import logging
try:
    import unittest.mock as mock
except ImportError:
    import mock
import tempfile
import shutil

logging.basicConfig(level=logging.DEBUG)
testlogger = logging.getLogger(__name__)


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


def get_mock_study():
    mock_algorithm = mock.MagicMock()
    mock_algorithm.get_suggestion.return_value = {'a': 1, 'b': 2}
    mock_stopping_rule = mock.MagicMock()

    return sherpa.Study(parameters=[sherpa.Discrete('a', [1,2]),
                                    sherpa.Choice('b', [2,5,7])],
                        algorithm=mock_algorithm,
                        stopping_rule=mock_stopping_rule,
                        lower_is_better=True,
                        disable_dashboard=True)


def get_test_trial(id=1):
    p = {'a': 1, 'b': 2}
    t = sherpa.Trial(id, p)
    return t

@pytest.fixture
def test_dir():
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)