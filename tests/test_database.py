import os
import pytest
import sys
import sherpa
import sherpa.core
import sherpa.schedulers
import sherpa.database
try:
    import unittest.mock as mock
except ImportError:
    import mock
import logging
import time
import warnings
from testing_utils import *


def test_database(test_dir):
    test_trial = get_test_trial()
    testlogger.debug(test_dir)
    db_port = sherpa.core._port_finder(27000, 28000)
    with sherpa.database._Database(test_dir, port=db_port) as db:
        time.sleep(2)
        testlogger.debug("Enqueuing...")
        db.enqueue_trial(test_trial)

        testlogger.debug("Starting Client...")
        client = sherpa.Client(port=db_port,
                               connectTimeoutMS=100,
                               serverSelectionTimeoutMS=1000)

        testlogger.debug("Getting Trial...")
        os.environ['SHERPA_TRIAL_ID'] = '1'
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

        # test that Sherpa raises correct error if MongoDB exits
        db2 = sherpa.database._Database(test_dir, port=db_port)
        with pytest.raises(OSError):
            db2.start()

            # with pytest.raises(RuntimeError):
            # with pytest.raises(pymongo.errors.ServerSelectionTimeoutError):
            #     client.get_trial()


def test_database_args(test_dir):
    custom_port = 26999
    testlogger.debug(test_dir)
    db_port = sherpa.core._port_finder(27000, 28000)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with sherpa.database._Database(test_dir, port=db_port,
                                       mongodb_args={
                                           "port": custom_port}) as db:
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert "Set port via the db_port" in str(w[-1].message)

            # test that there is something running on that port
            db2 = sherpa.database._Database(test_dir, port=custom_port)
            with pytest.raises(OSError):
                db2.start()


def test_client_test_mode_send_metrics_does_nothing():
    client = sherpa.Client(test_mode=True)
    trial = client.get_trial()

    assert trial.id == 1
    assert trial.parameters == {}

    client.send_metrics(trial=trial, iteration=1, objective=0.1)

@pytest.mark.skipif('keras' not in sys.modules,
                    reason="requires the Keras library")
def test_client_test_mode_keras_send_metrics_does_nothing():
    client = sherpa.Client(test_mode=True)
    trial = client.get_trial()

    assert trial.id == 1
    assert trial.parameters == {}

    callback = client.keras_send_metrics(trial=trial, objective_name='val_acc',
                                         context_names=['val_loss', 'loss',
                                                        'acc'])
