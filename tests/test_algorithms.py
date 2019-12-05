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
import pandas
import numpy
import sherpa
import logging
import itertools
import pytest
from testing_utils import *


def test_median_stopping_rule():
    results_df = pandas.DataFrame(collections.OrderedDict(
        [('Trial-ID', [1]*3 + [2]*3 + [3]*3),
         ('Status', ['INTERMEDIATE']*9),
         ('Iteration', [1, 2, 3]*3),
         ('a', [1, 1, 1]*3),
         ('b', [2, 2, 2]*3),
         ('Objective', [0.1]*3 + [0.2]*3 + [0.3]*3)]
    ))

    stopper = sherpa.algorithms.MedianStoppingRule(min_iterations=2,
                                                   min_trials=1)

    t = get_test_trial(id=3)

    assert stopper.should_trial_stop(trial=t, results=results_df,
                                     lower_is_better=True)

    stopper = sherpa.algorithms.MedianStoppingRule(min_iterations=4,
                                                   min_trials=1)
    assert not stopper.should_trial_stop(trial=t, results=results_df,
                                         lower_is_better=True)

    stopper = sherpa.algorithms.MedianStoppingRule(min_iterations=2,
                                                   min_trials=4)
    assert not stopper.should_trial_stop(trial=t, results=results_df,
                                         lower_is_better=True)


def get_local_search_study_lower_is_better(params, seed):
    alg = sherpa.algorithms.LocalSearch(seed_configuration=seed)

    study = sherpa.Study(parameters=params, algorithm=alg,
                         lower_is_better=True,
                         disable_dashboard=True)
    return study


class TestLocalSearch:
    @pytest.mark.parametrize("parameter,seed,expected",
                             [(sherpa.Ordinal('p', [0, 1, 2, 3, 4]), {'p': 2}, [1, 3]),
                              (sherpa.Continuous('p', [0, 1]), {'p': 0.5}, [0.5*0.8, 0.5*1.2]),
                              (sherpa.Discrete('p', [0, 10]), {'p': 5}, [4, 6]),
                              (sherpa.Choice('p', [0, 1, 2, 3, 4]), {'p': 2}, [0, 1, 3, 4])])
    def test_seed_and_first_suggestion(self, parameter, seed, expected):
        study = get_local_search_study_lower_is_better([parameter],
                                                       seed)
        trial = study.get_suggestion()
        assert trial.parameters['p'] == seed['p']
        study.add_observation(trial, objective=trial.parameters['p'], iteration=1)
        study.finalize(trial)

        trial = study.get_suggestion()
        assert trial.parameters['p'] in expected

    @pytest.mark.parametrize("parameter,seed,expected",
                             [(sherpa.Ordinal('p', [0, 1, 2, 3, 4]), {'p': 2}, [0, 1]),
                              (sherpa.Continuous('p', [0, 1]), {'p': 0.5}, [0.5*(0.8), 0.5*(0.8)**2]),
                              (sherpa.Discrete('p', [0, 10]), {'p': 5}, [int(5*(0.8)), int(5*(0.8)**2)]),
                              (sherpa.Choice('p', [0, 1, 2]), {'p': 2}, [0])])
    def test_expected_value_after_three_iterations(self, parameter, seed, expected):
        study = get_local_search_study_lower_is_better([parameter],
                                                       seed)
        for trial in study:
            study.add_observation(trial, objective=trial.parameters['p'], iteration=1)
            study.finalize(trial)
            if trial.id == 3:
                break

        assert study.get_best_result()['Objective'] in expected

    @pytest.mark.parametrize("param1,seed1,param2,seed2", [(sherpa.Ordinal('p1', [0, 1, 2, 3, 4]), {'p1': 2},
                                                            sherpa.Continuous('p2', [0, 1]), {'p2': 0.5})])
    def test_only_one_parameter_is_perturbed_at_a_time(self, param1, seed1, param2, seed2):
        seed = dict(seed1, **seed2)
        study = get_local_search_study_lower_is_better([param1, param2],
                                                       seed=seed)
        trial = study.get_suggestion()
        study.add_observation(trial, objective=1, iteration=1)
        study.finalize(trial)

        trial = study.get_suggestion()
        assert not all(
            param_value != seed[param_name] for param_name, param_value in
            trial.parameters.items())


def test_Iterate_search():
    '''
    The Iterate algorithm should be able to iterate over unhashable types.
    '''
    hp_iter = [{'a': 1, 'b': 'a', 'c': [10]},
               {'a': 1, 'b': 'a', 'c': [10]},
               {'a': 1, 'b': 'b', 'c': [10, 10]},
               {'a': 2, 'b': 'b', 'c': {'key':'value'}},
              ]
    alg = sherpa.algorithms.Iterate(hp_iter)
    parameters = alg.get_parameters()

    assert len(parameters) == 3
    assert set([len(p.range) for p in parameters]) == set([2,2,3])
    
    seen = []
    suggestion = alg.get_suggestion(parameters)
    while suggestion:
        seen.append((suggestion['a'], suggestion['b'], suggestion['c']))
        suggestion = alg.get_suggestion(parameters)

    assert seen == [(1, 'a', [10]), (1, 'a', [10]),
                    (1, 'b', [10,10]), (2, 'b', {'key':'value'})]


def test_grid_search():
    parameters = [sherpa.Choice('choice', ['a', 'b']),
                  sherpa.Continuous('continuous', [2, 3])]

    alg = sherpa.algorithms.GridSearch(num_grid_points=2)

    suggestion = alg.get_suggestion(parameters)
    seen = set()

    while suggestion != sherpa.AlgorithmState.DONE:
        seen.add((suggestion['choice'], suggestion['continuous']))
        suggestion = alg.get_suggestion(parameters)

    assert seen == {('a', 2.0),
                    ('a', 3.0),
                    ('b', 2.0),
                    ('b', 3.0)}


def test_grid_search_continuous():
    parameters = [sherpa.Continuous('continuous', [1, 3])]

    alg = sherpa.algorithms.GridSearch(num_grid_points=3)

    suggestion = alg.get_suggestion(parameters)
    seen = set()

    while suggestion != sherpa.AlgorithmState.DONE:
        seen.add(suggestion['continuous'])
        suggestion = alg.get_suggestion(parameters)

    assert seen == {1., 2., 3.}


def test_grid_search_log_continuous():
    parameters = [sherpa.Continuous('log-continuous', [1e-4,1e-2], 'log')]

    alg = sherpa.algorithms.GridSearch(num_grid_points=3)

    suggestion = alg.get_suggestion(parameters)
    seen = set()

    while suggestion != sherpa.AlgorithmState.DONE:
        seen.add(suggestion['log-continuous'])
        suggestion = alg.get_suggestion(parameters)

    assert seen == {1e-4, 1e-3, 1e-2}


def test_pbt():
    parameters = [sherpa.Continuous(name='param_a', range=[0, 1])]

    algorithm = sherpa.algorithms.PopulationBasedTraining(num_generations=3,
                                                          population_size=20,
                                                          parameter_range={'param_a': [0., 1.2]})

    study = sherpa.Study(parameters=parameters,
                         algorithm=algorithm,
                         lower_is_better=True,
                         disable_dashboard=True)

    for _ in range(20):
        trial = study.get_suggestion()
        print("Trial-ID={}".format(trial.id))
        print(trial.parameters)
        print()
        study.add_observation(trial=trial, iteration=1, objective=trial.id)
        study.finalize(trial=trial,
                       status='COMPLETED')

    for _ in range(16):
        trial = study.get_suggestion()
        print("Trial-ID={}".format(trial.id))
        print(trial.parameters)
        print()
        parent_param = study.results.loc[study.results['Trial-ID']==int(trial.parameters['load_from'])]['param_a'].iloc[0]
        print(parent_param)
        assert trial.parameters['param_a'] == parent_param
        assert trial.parameters['load_from'] == str(trial.id - 20)
        study.add_observation(trial=trial, iteration=1, objective=trial.id)
        study.finalize(trial=trial,
                       status='COMPLETED')

    for _ in range(4):
        trial = study.get_suggestion()
        print("Trial-ID={}".format(trial.id))
        print(trial.parameters)
        print()
        parent_param = study.results.loc[study.results['Trial-ID']==int(trial.parameters['load_from'])]['param_a'].iloc[0]
        print(parent_param)
        assert (trial.parameters['param_a'] == 0.8 * parent_param or
                trial.parameters['param_a'] == 1.0 * parent_param or
                trial.parameters['param_a'] == 1.2 * parent_param or
                trial.parameters['param_a'] == 0. or
                trial.parameters['param_a'] == 1.2)
        assert int(trial.parameters['load_from']) in [1, 2, 3, 4]
        study.add_observation(trial=trial, iteration=1, objective=trial.id)
        study.finalize(trial=trial,
                       status='COMPLETED')

    for _ in range(16):
        trial = study.get_suggestion()
        print("Trial-ID={}".format(trial.id))
        print(trial.parameters)
        print()
        parent_param = study.results.loc[study.results['Trial-ID']==int(trial.parameters['load_from'])]['param_a'].iloc[0]
        print(parent_param)
        assert trial.parameters['param_a'] == parent_param
        assert trial.parameters['load_from'] == str(trial.id - 20)
        study.add_observation(trial=trial, iteration=1, objective=trial.id)
        study.finalize(trial=trial,
                       status='COMPLETED')

    for _ in range(4):
        trial = study.get_suggestion()
        print("Trial-ID={}".format(trial.id))
        print(trial.parameters)
        print()
        parent_param = study.results.loc[study.results['Trial-ID']==int(trial.parameters['load_from'])]['param_a'].iloc[0]
        print(parent_param)
        assert (trial.parameters['param_a'] == 0.8 * parent_param or
                trial.parameters['param_a'] == 1.0 * parent_param or
                trial.parameters['param_a'] == 1.2 * parent_param or
                trial.parameters['param_a'] == 0. or
                trial.parameters['param_a'] == 1.2)
        assert int(trial.parameters['load_from']) in list(range(1, 9)) + list(range(21, 29))
        study.add_observation(trial=trial, iteration=1, objective=trial.id)
        study.finalize(trial=trial,
                       status='COMPLETED')

    assert study.get_suggestion() == sherpa.AlgorithmState.DONE


def test_pbt_ordinal():
    parameters = [sherpa.Ordinal(name='param_a', range=[-1, 0, 1])]

    algorithm = sherpa.algorithms.PopulationBasedTraining(num_generations=2,
                                                          population_size=10)

    study = sherpa.Study(parameters=parameters,
                         algorithm=algorithm,
                         lower_is_better=True,
                         disable_dashboard=True)

    for _ in range(10):
        trial = study.get_suggestion()
        print("Trial-ID={}".format(trial.id))
        print(trial.parameters)
        print()
        study.add_observation(trial=trial, iteration=1, objective=trial.parameters['param_a']*0.1)
        study.finalize(trial=trial,
                       status='COMPLETED')

    for _ in range(10):
        trial = study.get_suggestion()
        print("Trial-ID={}".format(trial.id))
        print(trial.parameters)
        print()
        assert trial.parameters['param_a'] in (-1, 0, 1)
        study.add_observation(trial=trial, iteration=1, objective=trial.parameters['param_a']*0.1)
        study.finalize(trial=trial,
                       status='COMPLETED')


def test_genetic():
    """
    Since genetic algorithms are stochastic we will check for average improvements while testing new configurations
    """
    parameters = [sherpa.Ordinal(name='param_a',
                                 range=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                  sherpa.Ordinal(name='param_b',
                                 range=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                                        'i', 'j']),
                  sherpa.Ordinal(name='param_c',
                                 range=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), ]

    algorithm = sherpa.algorithms.Genetic()

    study = sherpa.Study(parameters=parameters,
                         algorithm=algorithm,
                         lower_is_better=False,
                         disable_dashboard=True)
    mean_values = []
    for _ in range(500):
        results = study.results
        if results.shape[0] > 0:
            results = results[results['Status'] == 'COMPLETED']
            mean_values.append(results['Objective'].mean())
        trial = study.get_suggestion()
        print("Trial-ID={}".format(trial.id))
        print(trial.parameters)
        print()
        study.add_observation(trial=trial, iteration=1,
                              objective=trial.parameters['param_a'] * 0.1 +
                                        trial.parameters['param_c'] * 0.1)
        study.finalize(trial=trial,
                       status='COMPLETED')
    ascending = 0
    for pos in range(len(mean_values) - 1):
        if mean_values[pos + 1] > mean_values[pos]:
            ascending += 1
    print(ascending / len(mean_values))
    assert ascending / len(
        mean_values) > 0.7, "At least 70% of times we add a new result we must improve the average Objective"


def test_random_search():
    parameters = [sherpa.Continuous('a', [0, 1]),
                  sherpa.Choice('b', ['x', 'y', 'z'])]

    rs = sherpa.algorithms.RandomSearch(max_num_trials=10)
    last_config = {}

    for i in range(10):
        config = rs.get_suggestion(parameters=parameters)
        assert config != last_config
        last_config = config

    assert rs.get_suggestion(parameters=parameters) == sherpa.AlgorithmState.DONE


    rs = sherpa.algorithms.RandomSearch()
    last_config = {}

    for _ in range(1000):
        config = rs.get_suggestion(parameters=parameters)
        assert config != last_config
        last_config = config


def test_repeat_rs():
    parameters = [sherpa.Continuous('a', [0, 1]),
                  sherpa.Choice('b', ['x', 'y', 'z'])]
    rs = sherpa.algorithms.RandomSearch(max_num_trials=10)
    rs = sherpa.algorithms.Repeat(algorithm=rs, num_times=10)
    config_repeat = {}

    for i in range(10):
        config = rs.get_suggestion(parameters=parameters)
        assert config != config_repeat
        for j in range(9):
            config_repeat = rs.get_suggestion(parameters=parameters)
            assert config == config_repeat

    assert rs.get_suggestion(parameters=parameters) == sherpa.AlgorithmState.DONE


def test_repeat_grid_search():
    parameters = [sherpa.Choice('a', [1, 2]),
                  sherpa.Choice('b', ['a', 'b'])]

    alg = sherpa.algorithms.GridSearch()
    alg = sherpa.algorithms.Repeat(algorithm=alg, num_times=3)

    suggestion = alg.get_suggestion(parameters)
    seen = list()

    while suggestion != sherpa.AlgorithmState.DONE:
        seen.append((suggestion['a'], suggestion['b']))
        suggestion = alg.get_suggestion(parameters)

    expected_params = [(1, 'a'),
                       (1, 'b'),
                       (2, 'a'),
                       (2, 'b')]

    expected = list(itertools.chain.from_iterable(
        itertools.repeat(x, 3) for x in expected_params))

    print(sorted(expected))
    print(sorted(seen))

    assert sorted(expected) == sorted(seen)


def test_repeat_wait_for_completion():
    parameters = [sherpa.Continuous('a', [0, 1]),
                  sherpa.Choice('b', ['x', 'y', 'z'])]
    rs = sherpa.algorithms.RandomSearch(max_num_trials=10)
    rs = sherpa.algorithms.Repeat(algorithm=rs, num_times=10,
                                  wait_for_completion=True)
    study = sherpa.Study(parameters=parameters, algorithm=rs,
                         lower_is_better=True,
                         disable_dashboard=True)

    for i in range(10):
        tnew = study.get_suggestion()
        print(tnew.parameters)
        assert isinstance(tnew.parameters, dict)
        config = tnew.parameters
        study.add_observation(tnew, objective=float(i),
                              iteration=1)
        study.finalize(tnew)

        for j in range(9):
            t = study.get_suggestion()
            config_repeat = t.parameters
            assert config == config_repeat

            if j < 8:
                study.add_observation(t, objective=float(i),
                                      iteration=1)
                study.finalize(t)

        # Obtained 10/10 repeats for the configuration, but haven't added
        # results for the last one. Obtaining a new suggestion we expect WAIT.
        twait = study.get_suggestion()
        assert twait == sherpa.AlgorithmState.WAIT
        study.add_observation(t, objective=float(i),
                              iteration=1)
        study.finalize(t)

    tdone = study.get_suggestion()
    assert tdone == sherpa.AlgorithmState.DONE


def test_repeat_get_best_result():
    parameters = [sherpa.Choice('a', [1,2,3])]
    gs = sherpa.algorithms.GridSearch()
    gs = sherpa.algorithms.Repeat(algorithm=gs, num_times=3)
    study = sherpa.Study(parameters=parameters, algorithm=gs,
                         lower_is_better=True,
                         disable_dashboard=True)

    objectives = [1.1,1.2,1.3, 2.1,2.2,2.3, 9., 0.1, 9.1]

    for obj, trial in zip(objectives, study):
        study.add_observation(trial, objective=obj)
        study.finalize(trial)

    assert study.get_best_result()['a'] == 1  # not 3


def test_repeat_get_best_result_called_midway():
    parameters = [sherpa.Choice('a', [1,2,3])]
    gs = sherpa.algorithms.GridSearch()
    gs = sherpa.algorithms.Repeat(algorithm=gs, num_times=3)
    study = sherpa.Study(parameters=parameters, algorithm=gs,
                         lower_is_better=True,
                         disable_dashboard=True)

    objectives = [2.1,2.2,2.3, 9., 0.1, 9.1, 1.1,1.2,1.3]
    expected = [None, None, 1, 1, 1, 1, 1, 1, 3]

    for exp, obj, trial in zip(expected, objectives, study):
        study.add_observation(trial, objective=obj)
        study.finalize(trial)
        assert study.get_best_result().get('a') == exp



def test_repeat_results_aggregation():
    parameters = [sherpa.Continuous('myparam', [0, 1])]

    class MyAlg(sherpa.algorithms.Algorithm):
        allows_repetition = True
        def get_suggestion(self, parameters, results, lower_is_better):
            if results is not None and len(results) > 0:
                print(results)
                assert 'ObjectiveStdErr' in results.columns
                assert 'ObjectiveVar' in results.columns
                assert (results.loc[:, 'Objective'] == 0.).all()
                exp_std_err = numpy.sqrt(numpy.var([-1,0,1],ddof=1)/(3-1))
                assert (numpy.isclose(results.loc[:, 'ObjectiveStdErr'], exp_std_err)).all()
            return {'myparam': numpy.random.random()}


    alg = MyAlg()
    gs = sherpa.algorithms.Repeat(algorithm=alg,
                                  num_times=3,
                                  agg=True)
    study = sherpa.Study(algorithm=gs,
                         parameters=parameters,
                         lower_is_better=True,
                         disable_dashboard=True)
    for trial in study:
        study.add_observation(trial,
                              iteration=1,
                              objective=trial.id%3-1)  # 1->-1, 2->0, 3->1, 4->-1, ...
        study.finalize(trial)
        print(study.results)
        if trial.id > 10:
            break


def test_get_best_result():
    parameters = [sherpa.Choice('a', [1,2,3])]
    gs = sherpa.algorithms.GridSearch()
    study = sherpa.Study(parameters=parameters, algorithm=gs,
                         lower_is_better=True,
                         disable_dashboard=True)

    objectives = [1.1,1.2,1.3]

    for obj, trial in zip(objectives, study):
        study.add_observation(trial, objective=obj)
        study.finalize(trial)

    assert study.get_best_result()['a'] == 1


def test_chain():
    parameters = [sherpa.Continuous('a', [0, 1]),
                  sherpa.Choice('b', ['x', 'y', 'z'])]
    algorithm = sherpa.algorithms.Chain(algorithms=[sherpa.algorithms.GridSearch(num_grid_points=2),
                                                    sherpa.algorithms.RandomSearch(max_num_trials=10)])
    study = sherpa.Study(parameters=parameters, algorithm=algorithm,
                         lower_is_better=True,
                         disable_dashboard=True)

    for trial in study:
        if trial.id < 7:
            assert trial.parameters['a'] in [0, 1]
            assert trial.parameters['b'] == ['x', 'y', 'z'][trial.id%3-1]
        else:
            assert trial.parameters['a'] not in [0, 1]


