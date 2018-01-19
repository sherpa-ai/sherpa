
# SHERPA

## Installation
Clone into ```/your/path/``` from GitLab:
```
cd /your/path/
git clone git@gitlab.ics.uci.edu:uci-igb/sherpa.git
```

Add SHERPA and GPU_LOCK to Python-path in your profile:
```
export PYTHONPATH=$PYTHONPATH:/your/path/sherpa/
export PYTHONPATH=$PYTHONPATH:/extra/pjsadows0/libs/shared/gpu_lock/
```

Add MongoDB, DRMAA and SGE to your profile:
```
source /auto/igb-libs/linux/centos/6.x/x86_64/profiles/general
export DRMAA_LIBRARY_PATH=/opt/sge/lib/lx-amd64/libdrmaa.so
module load sge
```

Install dependencies:
```
cd /your/path/sherpa
pip install -e .
```

or

```
pip install pandas
pip install pymongo
pip install numpy
pip install scipy
pip install sklearn
pip install flask
pip install drmaa
pip install enum34  # if on < Python 3.4
```

## Environment
You should have an environment-profile that sets path variables and potentially loads a Python Virtual environment. All variable settings above should go into that profile. Note that an SGE job will not load your `.bashrc` so all necessary settings need to be in your profile.

## SGE
SGE required submit options. In Sherpa, those are defined as a string via the `submit_options` argument in the scheduler. To run jobs on the Arcus machines, typical submit options would be: 
```-N myScript -P arcus.p -q arcus.q -l hostname='(arcus-1|arcus-2|arcus-3)'```.
The `-N` option defines the name. To run from Arcus 5 to 9 you would set `-q arcus-ubuntu.q` and `hostname` with the relevant machines you want to run on. The SHERPA runner script can run from any Arcus machine.

## Example
You can run an example by doing:
```
cd /your/path/sherpa/examples/bianchini/
python runner.py --env <path/to/your/environment>
```

## Getting Started
An optimization in SHERPA consists of a trial-script and a runner-script. 

### Trial-script 
The trial-script trains your machine learning model with a given
parameter-configuration and sends metrics to SHERPA.
```
import sherpa
import time

client = sherpa.Client()
trial = client.get_trial()
```
THE script gets its parameter-configuration from the Sherpa Client.
```
# Model training
num_iterations = 10
for i in range(num_iterations):
    pseudo_objective = trial.parameters['param_a'] / float(i + 1) * trial.parameters['param_b']
    time.sleep(1)
    client.send_metrics(trial=trial, iteration=i+1,
                        objective=pseudo_objective)
    # print("Trial {} Iteration {}.".format(trial.id, i+1))
# print("Trial {} finished.".format(trial.id))
```
During training you use `send_metrics` every iteration to return objective
values to SHERPA.


### Runner-script
The runner-script defines the optimization and runs SHERPA. Parameters are
defined as a list:
```
import sherpa
parameters = [sherpa.Choice(name="param_a",
                            range=[1, 2, 3]),
              sherpa.Continuous(name="param_b",
                                range=[0, 1])]
```
Once you decided on your parameters and their ranges you can choose an optimization
algorithm:
```
algorithm = sherpa.algorithms.RandomSearch(max_num_trials=10)
```
Different schedulers allow to run an optimization on one machine or a cluster:
```
scheduler = sherpa.schedulers.LocalScheduler()
```
Once everything is set up you run it:
```
results = sherpa.optimize(parameters=parameters,
                          algorithm=algorithm,
                          lower_is_better=True,
                          filename=filename,
                          output_dir=tempdir,
                          scheduler=scheduler,
                          max_concurrent=2,
                          verbose=1)
```
The code for this example can be run as `python ./examples/runner_mode.py` from
the SHERPA root.

