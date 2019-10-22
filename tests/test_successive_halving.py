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
from __future__ import absolute_import
import collections
from sherpa.algorithms import successive_halving
from testing_utils import *


def test_no_stragglers_lower_is_better():
    """
    Basic scenario where everything is finished. In that case we expect
    ASHA to add 3 to rung0, promote 1 to rung1, add 3 to rung0, promote one to
    rung1, add 3 to rung0, promote 1 to rung1, promote 1 to rung2.
    """
    parameters = [sherpa.Continuous('myparam', [0, 1])]

    algorithm = successive_halving.SuccessiveHalving(r=1, R=9, eta=3, s=0,
                                                     max_finished_configs=1)

    study = sherpa.Study(parameters=parameters,
                         algorithm=algorithm,
                         lower_is_better=True,
                         disable_dashboard=True)

    rung_0_ids = [1, 2, 3, 5, 6, 7, 9, 10, 11]
    rung_1_ids = [4, 8, 12]
    rung_2_ids = [13]

    rung_0_promoted_ids = [1, 2, 3]
    rung_1_promoted_ids = [4]

    # Using trial.id as objective, hence the trials with the lowest trial ids
    # in the rung should get promoted, for rung 0 that is 1,2,3, for rung 1
    # 4

    for trial in study:
        if trial.id in rung_0_ids:
            assert trial.parameters['load_from'] == ''
            assert trial.parameters['save_to'] in map(str, rung_0_ids)
            assert trial.parameters['rung'] == 0
            assert trial.parameters['resource'] == 1

        elif trial.id in rung_1_ids:
            assert trial.parameters['load_from'] in map(str,
                                                        rung_0_promoted_ids)
            assert trial.parameters['save_to'] in map(str, rung_1_ids)
            assert trial.parameters['rung'] == 1
            assert trial.parameters['resource'] == 3

        elif trial.id in rung_2_ids:
            assert trial.parameters['load_from'] in map(str,
                                                        rung_1_promoted_ids)
            assert trial.parameters['save_to'] in map(str, rung_2_ids)
            assert trial.parameters['rung'] == 2
            assert trial.parameters['resource'] == 9

        study.add_observation(trial=trial, iteration=1,
                              objective=float(trial.id))
        study.finalize(trial)


def test_no_stragglers_larger_is_better():
    """
    Same as test_no_stragglers but with lower_is_better==False.
    """
    parameters = [sherpa.Continuous('myparam', [0, 1])]

    algorithm = successive_halving.SuccessiveHalving(r=1, R=9, eta=3, s=0,
                                                     max_finished_configs=1)

    study = sherpa.Study(parameters=parameters,
                         algorithm=algorithm,
                         lower_is_better=False,
                         disable_dashboard=True)

    rung_0_ids = [1, 2, 3, 5, 7]
    rung_1_ids = [4, 6, 8]
    rung_2_ids = [9]

    rung_0_promoted_ids = [3, 5, 7]
    rung_1_promoted_ids = [8]

    # Using trial.id as objective, hence the trials with the lowest trial ids
    # in the rung should get promoted, for rung 0 that is 1,2,3, for rung 1
    # 4

    for trial in study:
        print(trial.id, trial.parameters)
        if trial.id in rung_0_ids:
            assert trial.parameters['load_from'] == ''
            assert trial.parameters['save_to'] in map(str, rung_0_ids)
            assert trial.parameters['rung'] == 0
            assert trial.parameters['resource'] == 1

        elif trial.id in rung_1_ids:
            assert trial.parameters['load_from'] in map(str,
                                                        rung_0_promoted_ids)
            assert trial.parameters['save_to'] in map(str, rung_1_ids)
            assert trial.parameters['rung'] == 1
            assert trial.parameters['resource'] == 3

        elif trial.id in rung_2_ids:
            assert trial.parameters['load_from'] in map(str,
                                                        rung_1_promoted_ids)
            assert trial.parameters['save_to'] in map(str, rung_2_ids)
            assert trial.parameters['rung'] == 2
            assert trial.parameters['resource'] == 9

        study.add_observation(trial=trial, iteration=1,
                              objective=float(trial.id))
        study.finalize(trial)


def test_concurrent_evaluation():
    """
    Assume we were evaluating k trials in parallel, so when the first out of
    those k trials is finished there are still k-1 unfinished trials, i.e.
    only one result is available so far and nothing is promotable yet. Then
    we would expect ASHA to add another trial to the bottom rung as a next step.
    """
    parameters = [sherpa.Continuous('myparam', [0, 1])]

    algorithm = successive_halving.SuccessiveHalving(r=1, R=9, eta=3, s=0,
                                                     max_finished_configs=1)

    study = sherpa.Study(parameters=parameters,
                         algorithm=algorithm,
                         lower_is_better=True,
                         disable_dashboard=True)
    
    k = 6
    # Submit k trials
    running_trials = collections.deque()
    for _ in range(k):
        trial = study.get_suggestion()
        print(trial.id)
        assert trial.parameters['load_from'] == ''
        assert trial.parameters['rung'] == 0
        assert trial.parameters['save_to'] == str(trial.id)
        running_trials.append(trial)
        
    # First one finishes. Since no other trials have finished a new config
    # is added to the bottom rung
    print("Trial 1 finished")
    trial = running_trials.popleft()
    study.add_observation(trial=trial, iteration=1,
                          objective=float(trial.id))
    study.finalize(trial)
    trial = study.get_suggestion()
    assert trial.parameters['load_from'] == ''
    assert trial.parameters['rung'] == 0
    running_trials.append(trial)
    
    # Now two more finish. The first one gets added to the bottom rung, the
    # second one has enough completed configs to promote one to rung 1
    print("Trial 2 finished")
    trial = running_trials.popleft()
    study.add_observation(trial=trial, iteration=1,
                          objective=float(trial.id))
    study.finalize(trial)
    trial = study.get_suggestion()
    assert trial.parameters['load_from'] == ''
    assert trial.parameters['rung'] == 0
    running_trials.append(trial)

    print("Trial 3 finished")
    trial = running_trials.popleft()
    study.add_observation(trial=trial, iteration=1,
                          objective=float(trial.id))
    study.finalize(trial)
    trial = study.get_suggestion()
    assert trial.parameters['load_from'] == '1'
    assert trial.parameters['rung'] == 1
    running_trials.append(trial)

    # Now three more finish. Again the first two get added to the bottom rung
    # because there aren't enough completed unpromoted trials to promote one.
    print("Trial 4 finished")
    trial = running_trials.popleft()
    study.add_observation(trial=trial, iteration=1,
                          objective=float(trial.id))
    study.finalize(trial)
    trial = study.get_suggestion()
    print(trial.parameters)
    assert trial.parameters['load_from'] == ''
    assert trial.parameters['rung'] == 0
    running_trials.append(trial)

    print("Trial 5 finished")
    trial = running_trials.popleft()
    study.add_observation(trial=trial, iteration=1,
                          objective=float(trial.id))
    study.finalize(trial)
    trial = study.get_suggestion()
    assert trial.parameters['load_from'] == ''
    assert trial.parameters['rung'] == 0
    running_trials.append(trial)

    print("Trial 6 finished")
    trial = running_trials.popleft()
    study.add_observation(trial=trial, iteration=1,
                          objective=float(trial.id))
    study.finalize(trial)
    trial = study.get_suggestion()
    assert trial.parameters['load_from'] == '2'
    assert trial.parameters['rung'] == 1
    running_trials.append(trial)


def test_max_configs():
    """
    Make sure that ASHA stops when done.
    """
    parameters = [sherpa.Continuous('myparam', [0, 1])]

    algorithm = successive_halving.SuccessiveHalving(r=1, R=9, eta=3, s=0,
                                                     max_finished_configs=5)

    study = sherpa.Study(parameters=parameters,
                         algorithm=algorithm,
                         lower_is_better=False,
                         disable_dashboard=True)

    for trial in study:
        print(trial.id, trial.parameters)

        study.add_observation(trial=trial, iteration=1,
                              objective=float(trial.id))
        study.finalize(trial)

    completed = study.results.query("Status == 'COMPLETED'")

    assert len(completed.loc[completed.rung == 2, :]) == 5


if __name__ == '__main__':
    test_no_stragglers_lower_is_better()