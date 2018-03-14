
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
module load mongodb/2.6
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
SGE requires submit options. In Sherpa, those are defined as a string via the `submit_options` argument in the scheduler. To run jobs on the Arcus machines, typical submit options would be: 
```-N myScript -P arcus.p -q arcus_gpu.q -l hostname='(arcus-1|arcus-2|arcus-3)'```.
The `-N` option defines the name. The SHERPA runner script can run from any Arcus machine.

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
parameter-configuration and sends metrics to SHERPA. To get a trial:
```
import sherpa

client = sherpa.Client()
trial = client.get_trial()
```
The trial contains the parameter configuration for your training:
```
# Model training
num_iterations = 10
for i in range(num_iterations):
    pseudo_objective = trial.parameters['param_a'] / float(i + 1) * trial.parameters['param_b']
    client.send_metrics(trial=trial, iteration=i+1,
                        objective=pseudo_objective)
    # print("Trial {} Iteration {}.".format(trial.id, i+1))
# print("Trial {} finished.".format(trial.id))
```
During training `send_metrics` is used every iteration to return objective
values to SHERPA


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
Once you decided on the parameters and their ranges you can choose an optimization
algorithm:
```
algorithm = sherpa.algorithms.RandomSearch(max_num_trials=10)
```
Schedulers allow to run an optimization on one machine or a cluster:
```
scheduler = sherpa.schedulers.LocalScheduler()
```
The optimization is run via:
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


# Bayesian Optimization

## Background
Bayesian optimization for hyperparameter tuning as described for example by Snoek 
et al. 2012 uses a flexible model to map from hyperparameter space to objective 
values. In many cases this model is a Gaussian Process (GP). Given inputs of hyperparameter
configurations and outputs of objective values the GP can be fitted. This GP 
can then be used to make predictions about new hyperparameter configurations. 
Each prediction can then be evaluated with respect to its utility via an acquisiton function. 
The algorithm therefore consists of fitting the GP, finding the maximum of the 
acquisition function, evaluating the chosen hyperparameter configuration, and repeating the process.
This yields an optimization method that under certain conditions can be proved to be 
consistent.

## SHERPA Implementation
SHERPA implements Bayesian optimization using a GP and an Expected Improvement
acquisition function. In contrast to Snoek et al. 2012, SHERPA obtains the 
hyperparameters of the Gaussian Process via maximization of the marginal likelihood. 
At this point there is no thorough treatment of parallel evaluation in place. However,
in practice this is not an issue since the optimium of the acquistion function is
found via random sampling and numerical optimization of the best samples. It is 
therefore unlikely that the same configuration is suggested twice. SHERPA's 
`Discrete` parameter is treated like a continuous variable that is discretized after 
a value is suggested. `Choice` parameters are treated as categorical/one-hot variables 
in the GP as justified by Duvenaud's Kernel Cookbook.

> Snoek, Jasper, Hugo Larochelle, and Ryan P. Adams. "Practical bayesian optimization of machine learning algorithms." Advances in neural information processing systems. 2012.

> https://www.cs.toronto.edu/~duvenaud/cookbook/

# Population Based Training

## Background
Population Based Training (PBT) as introduced by Jaderberg et al. 2017 is an evolutionary
type algorithm. The figure below shows a diagram of what PBT is doing. PBT starts
with a random population of hyperparamater configurations. Each population member
is trained for a limited amount of time and evaluated. When every population member
has been evaluated, the ones with low scores replace their own weights and 
hyperparameters with those from population members with high scores (exploit) and 
perturb the hyperparameters (explore). Then all population members are trained and evaluated
again and the process repeats.

![PBT diagram](pbt.png "PBT Diagram from Jaderberg et al. 2017")

## SHERPA Implementation
SHERPA implements this algorithm as follows. A population of hyperparameter 
configurations is trained and evaluated for an amount of time implicityly 
specified by the user through the Trial-script. Population members are then 
sampled from the top third of the population. For each sampled member each 
hyperparameter is randomly increased, decreased, or held constant. Population 
members are always sampled from the generation previous to the current one.

## Example

### Trial-Script
In order to use PBT, the Trial-script needs to implement some additional functionality
as compared to the regular Trial-script. The parameters are obtained as usual:
```
client = sherpa.Client()
trial = client.get_trial() 
```

#### Load and Perturb
The `trial.parameters` will now also contain the keys `load_from`, `save_to`, and
`lineage`. The `lineage` indicates the heritage of this trial in terms of trial
IDs and can be ignored at this point. The `load_from` key indicates whether weights 
need to be loaded. For example in Keras:
```
if trial.parameters['load_from'] == '':
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=trial.parameters['lr'],
                                                 momentum=trial.parameters['momentum']),
                  metrics=['accuracy'])
else:
    model = load_model(os.path.join('./output', trial.parameters['load_from']))
    K.set_value(model.optimizer.lr, trial.parameters['lr'])
    K.set_value(model.optimizer.momentum, trial.parameters['momentum'])
```
Note that if the model is loaded then the user may have to change the hyperparameters 
manually for compiled models.

#### Save
After the model is trained and evaluated it is crucial that it is saved to 
`save_to`. The user can choose where to save the models to and what exact name 
to give them so long each is identifiable by the number given in `save_to`.
```
model.save(os.path.join('./output', trial.parameters['save_to']))
```

### Runner Script
The runner script is as usual. The parameters for the PBT algorithm are 
population size, and parameter range. Population size is the number of models 
that are randomly initialized at the beginning and the size of every generation 
thereafter. The parameter ranges correspond to ranges used by PBT for perturbation.
The motivation for this parameter is that one may want the initial models to be 
sampled from the ranges provided in the regular way. The PBT parameter ranges if 
the space that hyperparameters need to stay in via perturbation.

```
pbt_ranges = {'lr':[0.0000001, 1.], 'batch_size':[16, 32, 64, 128]}
algorithm = sherpa.algorithms.PopulationBasedTraining(population_size=50,
                                                      parameter_range=pbt_ranges)
```

>Jaderberg, Max, et al. "Population Based Training of Neural Networks." arXiv preprint arXiv:1711.09846 (2017).
