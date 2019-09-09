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
import sherpa
import time

parameters = [sherpa.Choice(name="param_a",
                            range=[1, 2, 3]),
              sherpa.Continuous(name="param_b",
                                range=[0, 1])]

algorithm = sherpa.algorithms.RandomSearch(max_num_trials=40)
# algorithm = sherpa.algorithms.LocalSearch(num_random_seeds=20)
# algorithm = sherpa.algorithms.BayesianOptimization(num_grid_points=2, max_num_trials=50)
# stopping_rule = sherpa.algorithms.MedianStoppingRule(min_iterations=2,
#                                           min_trials=5)
stopping_rule = None
study = sherpa.Study(parameters=parameters,
                     algorithm=algorithm,
                     stopping_rule=stopping_rule,
                     lower_is_better=True,
                     dashboard_port=8999)

num_iterations = 10

# get trials from study by iterating or study.get_suggestion()
for trial in study:
    print("Trial {}:\t{}".format(trial.id, trial.parameters))

    # Simulate model training
    for i in range(num_iterations):
        
        # access parameters via trial.parameters and id via trial.id
        pseudo_objective = trial.parameters['param_a'] / float(i + 1) * trial.parameters['param_b']
        
        # add observations once or multiple times
        study.add_observation(trial=trial,
                              iteration=i+1,
                              objective=pseudo_objective,
                              context={'pseudo_acc': 1-pseudo_objective})
        time.sleep(1)

        if study.should_trial_stop(trial=trial):
            print("Stopping Trial {} after {} iterations.".format(trial.id, i+1))
            study.finalize(trial=trial,
                           status='STOPPED')
            break
    else:
        study.finalize(trial=trial,
                       status='COMPLETED')

print(study.get_best_result())
