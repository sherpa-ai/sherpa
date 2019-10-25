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
#from spacetime import Node, Dataframe
import collections
import os

import pandas
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest

import sherpa
import sherpa.core
import sherpa.data_collection.spacetime_database
import sherpa.schedulers

try:
    import unittest.mock as mock
except ImportError:
    import mock
import logging
import tempfile
import shutil
import time
import warnings

logging.basicConfig(level=logging.DEBUG)
testlogger = logging.getLogger(__name__)

def test_trial():
    p = {'a': 1, 'b': 2}
    t = sherpa.Trial(1, p)
    assert t.id == 1
    assert t.parameters == p

def get_test_trial(id=1):
    p = {'a': 1, 'b': 2}
    t = sherpa.Trial(id, p)
    return t

@pytest.fixture
def test_dir():
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)

def test_spacetime_data_collection(test_dir):
    test_trial = get_test_trial()
    testlogger.debug(test_dir)
    db_port = sherpa.core._port_finder(27000, 28000)
    with sherpa.data_collection.spacetime_database.SpacetimeServer(port=db_port) as db:
        time.sleep(2)
        testlogger.debug("Enqueuing...")

        db.enqueue_trial_results(test_trial)

        testlogger.debug("Starting Client...")

        client = sherpa.data_collection.spacetime_database.Client(port=db_port)

        testlogger.debug("Getting Trial...")
        os.environ['SHERPA_TRIAL_ID'] = '1'

        trial = client.get_trial()


        assert trial.id == 1
        assert  trial.parameters == {'a': 1, 'b': 2}

        testlogger.debug("Sending Metrics...")
        client.send_metrics(trial=trial, iteration=1,
                           objective=0.1, context={'other_metric': 0.2})
        new_results = db.get_new_results()
        testlogger.debug(new_results)
        assert new_results == [[('parameters', [('a', 1), ('b', 2)]),
                                ('trial_id', 1),
                                ('objective', 0.1),
                                ('iteration', 1),
                                ('context', [('other_metric', 0.2)]),
                                ('result_id', 1)]]
        new_results = db.get_new_results()
        assert new_results == None

        client.quit()
