from __future__ import absolute_import
import collections
import pandas
import sherpa
import logging
from test_sherpa import get_test_trial


logging.basicConfig(level=logging.DEBUG)
testlogger = logging.getLogger(__name__)


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


def test_local_search():
    parameters, results, lower_is_better = sherpa.algorithms.get_sample_results_and_params()

    previous_configs = [{'param_a': row['param_a'], 'param_b': row['param_b']}
                        for _, row in results.iterrows()]

    best_params = results.loc[42, ['param_a', 'param_b']].to_dict()

    num_random_seeds = 3
    num_seeds_configs = 5
    num_test_steps = 10

    rs = sherpa.algorithms.RandomSearch(num_random_seeds)

    seed_configs = [rs.get_suggestion(parameters, results, lower_is_better)
                    for _ in range(num_seeds_configs)]

    alg = sherpa.algorithms.LocalSearch(num_random_seeds=num_random_seeds,
                                        seed_configurations=seed_configs)

    # one parameter is continuous so we can assume we don't sample same value
    # twice

    # seed configs
    for seed in seed_configs:
        p = alg.get_suggestion(parameters, results, lower_is_better)

        assert p == seed

    # random seeds
    for _ in range(num_random_seeds):
        p = alg.get_suggestion(parameters, results, lower_is_better)

        assert p not in seed_configs
        assert p not in previous_configs

    # hill climb
    for _ in range(num_test_steps):
        p = alg.get_suggestion(parameters, results, lower_is_better)

        assert (best_params['param_a'] == p['param_a']
                or best_params['param_b'] == p['param_b'])


def test_grid_search():
    parameters = [sherpa.Choice('a', [1, 2]),
                  sherpa.Choice('b', ['a', 'b']),
                  sherpa.Continuous('c', [1, 4])]

    alg = sherpa.algorithms.GridSearch()

    suggestion = alg.get_suggestion(parameters)
    seen = set()

    while suggestion:
        seen.add((suggestion['a'], suggestion['b'], suggestion['c']))
        suggestion = alg.get_suggestion(parameters)

    assert seen == {(1, 'a', 2), (1, 'a', 3),
                    (1, 'b', 2), (1, 'b', 3),
                    (2, 'a', 2), (2, 'a', 3),
                    (2, 'b', 2), (2, 'b', 3)}


def test_grid_search_grid_maker():
    alg = sherpa.algorithms.GridSearch()

    assert alg._get_param_dict([sherpa.Continuous('b', [1, 4])]) == {'b': [2, 3]}
    assert alg._get_param_dict([sherpa.Continuous('c', [0.0001, 0.1], 'log')]) == {'c': [0.001, 0.01]}


def test_gp_ei_seeds():
    alg = sherpa.algorithms.BayesianOptimization(num_grid_points=2)

    parameters = sherpa.Parameter.grid({'a': [1, 2],
                                        'b': ['a', 'b']})
    parameters.append(sherpa.Continuous(name='c', range=[1, 4]))

    left = {(1, 'a', 2), (1, 'a', 3),
            (1, 'b', 2), (1, 'b', 3),
            (2, 'a', 2), (2, 'a', 3),
            (2, 'b', 2), (2, 'b', 3)}

    for _ in range(8):
        suggestion = alg.get_suggestion(parameters, None, True)
        print(suggestion)
        left.remove((suggestion['a'], suggestion['b'], suggestion['c']))

    assert len(left) == 0


def test_gp_ei():
    parameters = [sherpa.Choice(name="param_a",
                                range=[1, 2, 3]),
                  sherpa.Continuous(name="param_b",
                                    range=[0, 1])]

    algorithm = sherpa.algorithms.BayesianOptimization(num_grid_points=3)

    study = sherpa.Study(parameters=parameters,
                         algorithm=algorithm,
                         lower_is_better=True,
                         disable_dashboard=True)

    for trial in study:
        if trial.id == 50:
            break
        print("Trial {}:\t{}".format(trial.id, trial.parameters))

        pseudo_objective = trial.parameters['param_a'] * trial.parameters['param_b']

        study.add_observation(trial=trial,
                              iteration=1,
                              objective=pseudo_objective)
        study.finalize(trial=trial,
                       status='COMPLETED')

    rval = study.get_best_result()
    assert rval['param_a'] == 1.
    assert rval['param_b'] < 0.001


def test_pbt():
    parameters = [sherpa.Continuous(name='param_a', range=[0, 1])]

    algorithm = sherpa.algorithms.PopulationBasedTraining(population_size=20,
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
        study.add_observation(trial=trial, iteration=1, objective=trial.id*0.1)
        study.finalize(trial=trial,
                       status='COMPLETED')

    for _ in range(20):
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
        assert int(trial.parameters['load_from']) <= 10
        study.add_observation(trial=trial, iteration=1, objective=trial.id*0.1)
        study.finalize(trial=trial,
                       status='COMPLETED')

    for _ in range(20):
        trial = study.get_suggestion()
        print("Trial-ID={}".format(trial.id))
        print(trial.parameters)
        print()
        parent_param = study.results.loc[
            study.results['Trial-ID'] == int(trial.parameters['load_from'])][
            'param_a'].iloc[0]
        assert (trial.parameters['param_a'] == 0.8 * parent_param or
                trial.parameters['param_a'] == 1.0 * parent_param or
                trial.parameters['param_a'] == 1.2 * parent_param or
                trial.parameters['param_a'] == 0. or
                trial.parameters['param_a'] == 1.2)
        # assert int(trial.parameters['load_from']) <= 27
        study.add_observation(trial=trial, iteration=1, objective=trial.id*0.1)
        study.finalize(trial=trial,
                       status='COMPLETED')


def test_pbt_ordinal():
    parameters = [sherpa.Ordinal(name='param_a', range=[-1, 0, 1])]

    algorithm = sherpa.algorithms.PopulationBasedTraining(population_size=10)

    study = sherpa.Study(parameters=parameters,
                         algorithm=algorithm,
                         lower_is_better=True,
                         disable_dashboard=True)

    for _ in range(20):
        trial = study.get_suggestion()
        print("Trial-ID={}".format(trial.id))
        print(trial.parameters)
        print()
        study.add_observation(trial=trial, iteration=1, objective=trial.parameters['param_a']*0.1)
        study.finalize(trial=trial,
                       status='COMPLETED')

    for _ in range(20):
        trial = study.get_suggestion()
        print("Trial-ID={}".format(trial.id))
        print(trial.parameters)
        print()
        assert trial.parameters['param_a'] in (-1, 0, 1)
        study.add_observation(trial=trial, iteration=1, objective=trial.parameters['param_a']*0.1)
        study.finalize(trial=trial,
                       status='COMPLETED')