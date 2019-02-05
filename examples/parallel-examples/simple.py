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
import tempfile
import os
import sherpa
import sherpa.schedulers
import argparse
import socket
import sherpa.algorithms.bayesian_optimization as bayesian_optimization


parser = argparse.ArgumentParser()
parser.add_argument('--env', help='Your environment path.',
                    default='/home/lhertel/profiles/python3env.profile', type=str)
FLAGS = parser.parse_args()
# figuring out host and queue
host = socket.gethostname()
sge_q = 'arcus.q' if (host.startswith('arcus-1') or host.startswith('arcus-2') or host.startswith('arcus-3') or host.startswith('arcus-4')) else 'arcus-ubuntu.q'

tempdir = tempfile.mkdtemp(dir=".")

parameters = [sherpa.Choice(name="param_a",
                            range=[1, 2, 3]),
              sherpa.Continuous(name="param_b",
                                range=[0, 1])]


algorithm = sherpa.algorithms.RandomSearch(max_num_trials=10)
# stopping_rule = sherpa.algorithms.MedianStoppingRule(min_iterations=2,
#                                           min_trials=3)
# algorithm = bayesian_optimization.GPyOpt(max_concurrent=4,
#                                          model_type='GP',
#                                          acquisition_type='EI',
#                                          max_num_trials=100)

# scheduler = sherpa.schedulers.SGEScheduler(submit_options="-N example -P arcus.p -q {} -l hostname='{}'".format(sge_q, host), environment=FLAGS.env, output_dir=tempdir)

scheduler = sherpa.schedulers.LocalScheduler()

### The *training script*
testscript = """import sherpa
import time

client = sherpa.Client()
trial = client.get_trial()

# Simulate model training
num_iterations = 10
for i in range(num_iterations):
    pseudo_objective = trial.parameters['param_a'] / float(i + 1) * trial.parameters['param_b']
    time.sleep(1)
    client.send_metrics(trial=trial, iteration=i+1,
                        objective=pseudo_objective)
    # print("Trial {} Iteration {}.".format(trial.id, i+1))
# print("Trial {} finished.".format(trial.id))
"""

filename = os.path.join(tempdir, "test.py")
with open(filename, 'w') as f:
    f.write(testscript)

results = sherpa.optimize(parameters=parameters,
                          algorithm=algorithm,
                          lower_is_better=True,
                          filename=filename,
                          output_dir=tempdir,
                          scheduler=scheduler,
                          max_concurrent=4,
                          verbose=1)

print(results)
