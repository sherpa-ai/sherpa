from __future__ import print_function
import tempfile
import os
import sherpa
import sherpa.schedulers

tempdir = tempfile.mkdtemp(dir=".")

parameters = [sherpa.Choice(name="param_a",
                            range=[1, 2, 3]),
              sherpa.Continuous(name="param_b",
                                range=[0, 1])]

algorithm = sherpa.RandomSearch(max_num_trials=10)
stopping_rule = sherpa.MedianStoppingRule(min_iterations=2,
                                          min_trials=5)
study = sherpa.Study(parameters=parameters,
                     algorithm=algorithm,
                     lower_is_better=True,
                     dashboard_port=8999)

# scheduler = sherpa.schedulers.SGEScheduler(submit_options="-N example -P arcus.p -q arcus.q -l hostname='arcus-1'", environment="/home/lhertel/profiles/main.profile",
#                                            output_dir=tempdir)
# hostname = 'nimbus.ics.uci.edu'
db_port = 27010
scheduler = sherpa.schedulers.LocalScheduler()

### The *training script*
testscript = """import sherpa
import time

# client = sherpa.Client(host='nimbus.ics.uci.edu', port=28282)
client = sherpa.Client(host='localhost')
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

results = sherpa.optimize(filename=filename,
                          study=study,
                          output_dir=tempdir,
                          scheduler=scheduler,
                          max_concurrent=2,
                          db_port=db_port)

print(results)