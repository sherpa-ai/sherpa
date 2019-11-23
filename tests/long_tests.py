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
from __future__ import print_function
import os
import pytest
import sherpa
import sherpa.schedulers
from testing_utils import *


### The *training script*
testscript = """import sherpa
client = sherpa.Client(host='localhost', port=27002, serverSelectionTimeoutMS=1000)
trial = client.get_trial()

# Simulate model training
num_iterations = 10
for i in range(num_iterations):
    pseudo_objective = trial.parameters['param_a'] / float(i + 1) * trial.parameters['param_b']
    # time.sleep(1)
    client.send_metrics(trial=trial, iteration=i+1,
                        objective=pseudo_objective)
"""

# @pytest.mark.skipif(shutil.which('mongod') is None or 'TRAVIS' in os.environ,
#                     reason="requires MongoDB")
@pytest.mark.skip(reason="find out why it fails")
def test_wrong_db_host_or_port(test_dir):
    print("MONGODB: ", shutil.which('mongod'))
    tempdir = test_dir

    parameters = [sherpa.Choice(name="param_a",
                                range=[1, 2, 3]),
                  sherpa.Continuous(name="param_b",
                                    range=[0, 1])]

    algorithm = sherpa.algorithms.RandomSearch(max_num_trials=3)

    db_port = 27000
    scheduler = sherpa.schedulers.LocalScheduler()

    filename = os.path.join(tempdir, "test.py")
    with open(filename, 'w') as f:
        f.write(testscript)

    with pytest.warns(RuntimeWarning):
        results = sherpa.optimize(filename=filename,
                                  lower_is_better=True,
                                  algorithm=algorithm,
                                  parameters=parameters,
                                  output_dir=tempdir,
                                  scheduler=scheduler,
                                  max_concurrent=1,
                                  db_port=db_port)


### The *training script*
testscript2 = """import sherpa
client = sherpa.Client(host='localhost', port=27000, serverSelectionTimeoutMS=1000)
trial = client.get_trial()

1/0
"""

# @pytest.mark.skipif(shutil.which('mongod') is None or 'TRAVIS' in os.environ,
#                     reason="requires MongoDB")
@pytest.mark.skip(reason="find out why it fails")
def test_user_code_fails(test_dir):

    tempdir = test_dir

    parameters = [sherpa.Choice(name="param_a",
                                range=[1, 2, 3]),
                  sherpa.Continuous(name="param_b",
                                    range=[0, 1])]

    algorithm = sherpa.algorithms.RandomSearch(max_num_trials=3)

    db_port = 27000
    scheduler = sherpa.schedulers.LocalScheduler()

    filename = os.path.join(tempdir, "test.py")
    with open(filename, 'w') as f:
        f.write(testscript2)

    with pytest.warns(RuntimeWarning):
        results = sherpa.optimize(filename=filename,
                                  lower_is_better=True,
                                  algorithm=algorithm,
                                  parameters=parameters,
                                  output_dir=tempdir,
                                  scheduler=scheduler,
                                  max_concurrent=1,
                                  db_port=db_port)