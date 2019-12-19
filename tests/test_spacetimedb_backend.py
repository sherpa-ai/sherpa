import os
import pytest
import sys
import sherpa
import sherpa.core
import sherpa.schedulers
from sherpa.data_collection import SpaceTimeDBBackend, SpaceTimeDBClient
try:
    import unittest.mock as mock
except ImportError:
    import mock
import logging
import time
import warnings
from testing_utils import *

logging.basicConfig(level=logging.DEBUG)
testlogger = logging.getLogger(__name__)


def test_database(test_dir):
    test_trial = get_test_trial()
    test_trial2 = get_test_trial(2)
    testlogger.debug(test_dir)
    # db_port = sherpa.core._port_finder(27000, 28000)
    db_port = 27010
    with SpaceTimeDBBackend(test_dir, port=db_port) as db:
        time.sleep(2)
        testlogger.debug("Enqueuing...")
        db.enqueue_trial(test_trial)
        db.enqueue_trial(test_trial2)

        testlogger.debug("Starting Client...")
        client = SpaceTimeDBClient(port=db_port,
                               connectTimeoutMS=100,
                               serverSelectionTimeoutMS=1000)
        client1 = SpaceTimeDBClient(port=db_port,
                               connectTimeoutMS=100,
                               serverSelectionTimeoutMS=1000)

        testlogger.debug("Getting Trial...")
        os.environ['SHERPA_TRIAL_ID'] = '1'
        t = client.get_trial()
        assert t.id == 1
        assert t.parameters == {'a': 1, 'b': 2}

        # Testing get_trial for a second client
        os.environ['SHERPA_TRIAL_ID'] = '2'

        trial1 = client1.get_trial()

        assert trial1.id == 2
        assert  trial1.parameters == {'a': 1, 'b': 2}

        testlogger.debug("Sending Metrics...")
        client.send_metrics(trial=t, iteration=1,
                            objective=0.1, context={'other_metric': 0.2})

        new_results = db.get_new_results()
        testlogger.debug(new_results)
        assert new_results == [{'context': {'other_metric': 0.2},
                                'iteration': 1,
                                'objective': 0.1,
                                'parameters': {'a': 1, 'b': 2},
                                'trial_id': 1,
                                'result_id': 1}]
        # Test if there is no new results, get_new_results() should return an empty list []
        new_results = db.get_new_results()
        assert new_results == []

        # Test for a second submission
        client.send_metrics(trial = t, iteration = 2,objective = 0.01, context={'other_metric':2})
        new_results = db.get_new_results()

        assert new_results == [{'context': {'other_metric': 2},
                                'iteration': 2,
                                'objective': 0.01,
                                'parameters': {'a': 1, 'b': 2},
                                'trial_id': 1,
                                'result_id': 2}]

        # Test clinet1
        client1.send_metrics(trial=trial1, iteration=1,
                           objective=0.1, context={'other_metric': 0.2})
        new_results = db.get_new_results()
        testlogger.debug(new_results)

        assert new_results == [{'context': {'other_metric': 0.2},
                                'iteration': 1,
                                'objective': 0.1,
                                'parameters': {'a': 1, 'b': 2},
                                'trial_id': 2,
                                'result_id': 1}]


def test_client_test_mode_send_metrics_does_nothing():
    client = SpaceTimeDBClient(test_mode=True)
    trial = client.get_trial()

    assert trial.id == 1
    assert trial.parameters == {}

    client.send_metrics(trial=trial, iteration=1, objective=0.1)
