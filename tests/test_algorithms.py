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
    parameters = [sherpa.Continuous('cont', [0, 1]),
                  sherpa.Ordinal('ord', [1, 2, 3])]

    seed = {'cont': 0.1, 'ord': 3}
    alg = sherpa.algorithms.LocalSearch(seed_configuration=seed)

    study = sherpa.Study(parameters=parameters, algorithm=alg, lower_is_better=True,
                         disable_dashboard=True)

    def mock_objective(p):
        return p['cont']/p['ord']

    t = study.get_suggestion()
    assert t.parameters == seed
    study.add_observation(t, objective=mock_objective(t.parameters),
                          iteration=1)
    study.finalize(t)

    t = study.get_suggestion()
    t.parameters['cont'] = round(t.parameters['cont'], 5)
    print(t.parameters)
    assert t.parameters in [{'cont': 0.09, 'ord': 3}, {'cont': 0.1, 'ord': 2},
                            {'cont': 0.11, 'ord': 3}]
    study.add_observation(t, objective=mock_objective(t.parameters),
                          iteration=1)
    study.finalize(t)

    if t.parameters == {'cont': 0.09, 'ord': 3}:
        t = study.get_suggestion()
        t.parameters['cont'] = round(t.parameters['cont'], 5)
        assert t.parameters in [{'cont': 0.081, 'ord': 3},
                                {'cont': 0.099, 'ord': 3},
                                {'cont': 0.09, 'ord': 2}]
    else:
        t = study.get_suggestion()
        t.parameters['cont'] = round(t.parameters['cont'], 5)
        assert t.parameters in [{'cont': 0.09, 'ord': 3},
                                {'cont': 0.1, 'ord': 2},
                                {'cont': 0.11, 'ord': 3}]




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