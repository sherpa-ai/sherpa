from __future__ import print_function
import tempfile
import shutil
import os
import pytest
import sherpa
import sherpa.schedulers
from test_sherpa import test_dir


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


def test_wrong_db_host_or_port(test_dir):

    tempdir = test_dir

    parameters = [sherpa.Choice(name="param_a",
                                range=[1, 2, 3]),
                  sherpa.Continuous(name="param_b",
                                    range=[0, 1])]

    algorithm = sherpa.algorithms.RandomSearch(max_num_trials=3)

    study = sherpa.Study(parameters=parameters,
                         algorithm=algorithm,
                         lower_is_better=True)
    db_port = 27000
    scheduler = sherpa.schedulers.LocalScheduler()

    filename = os.path.join(tempdir, "test.py")
    with open(filename, 'w') as f:
        f.write(testscript)

    with pytest.warns(RuntimeWarning):
        results = sherpa.optimize(filename=filename,
                                  study=study,
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


def test_user_code_fails(test_dir):

    tempdir = test_dir

    parameters = [sherpa.Choice(name="param_a",
                                range=[1, 2, 3]),
                  sherpa.Continuous(name="param_b",
                                    range=[0, 1])]

    algorithm = sherpa.algorithms.RandomSearch(max_num_trials=3)

    study = sherpa.Study(parameters=parameters,
                         algorithm=algorithm,
                         lower_is_better=True)
    db_port = 27000
    scheduler = sherpa.schedulers.LocalScheduler()

    filename = os.path.join(tempdir, "test.py")
    with open(filename, 'w') as f:
        f.write(testscript2)

    with pytest.warns(RuntimeWarning):
        results = sherpa.optimize(filename=filename,
                                  study=study,
                                  output_dir=tempdir,
                                  scheduler=scheduler,
                                  max_concurrent=1,
                                  db_port=db_port)