
# SHERPA

Welcome to SHERPA - a hyperparameter tuning framework for machine learning.
In order to get SHERPA running clone the repository from GitLab by
calling ```git clone git@gitlab.ics.uci.edu:uci-igb/sherpa.git``` from the
command line and adding the directory to the Python path (e.g.
```export PYTHONPATH=$PYTHONPATH:/user/local/sherpa/```). In order to get the
necessary dependencies you can run ```python setup.py install``` from the SHERPA folder.

### Optional Dependencies
+ Drmaa 0.7.8 (for SGE)
+ Keras (for examples)
+ GPU Lock (for examples and recommended for SGE)

## Running Sherpa

### Parameters
Base class for a parameter.

### Algorithm
Abstract algorithm that returns next parameters conditional on parameter
ranges and previous results.

### Stopping Rules
Abstract class to evaluate whether a trial should stop conditional on all
results so far.

### Combining these into a Study
A Study defines an entire optimization and its results.

__Attributes:__

- __algorithm__ _(sherpa.algorithms.Algorithm)_: takes results table and returns
parameter set.
- __results__ _(pandas.DataFrame)_: contains results from this study.
- __parameters__ _(list[sherpa.Parameter])_: parameters being optimized.
- __stopping_rule__ _(sherpa.stopping_rules.StoppingRule)_: rule for stopping
trials prematurely.
- __lower_is_better__ _(bool)_: whether lower objective values are better.


### Scheduler
Abstract interface to a job scheduler.

The job scheduler gives an API to submit jobs and retrieve statuses of all
jobs.

### Putting it all together
Runs a Study via the Runner class.

__Arguments:__

- __filename__ _(str)_: the name of the file which is called to evaluate
configurations
- __study__ _(sherpa.Study)_: the Study to be run.
- __output_dir__ _(str)_: where scheduler and database files will be stored.
- __scheduler__ _(sherpa.Scheduler)_: a scheduler.
- __max_concurrent__ _(int)_: the number of trials that will be evaluated in
parallel.

