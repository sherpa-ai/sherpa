from __future__ import print_function
import tempfile
import os
import sherpa
import sherpa.schedulers
import argparse
import socket

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
                          max_concurrent=2,
                          verbose=1)

print(results)
