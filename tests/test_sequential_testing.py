import collections
import sherpa
import pandas
import tempfile
import shutil
import os
import numpy
from sherpa.algorithms import SequentialTesting, Algorithm
from sherpa.algorithms.bayesian_optimization import GPyOpt


def test_is_stage_done():
    results_df = pandas.DataFrame(collections.OrderedDict(
        [('Trial-ID', [1] * 1 + [2] * 1 + [3] * 1),
         ('Status', [sherpa.TrialStatus.COMPLETED] * 3),
         ('stage', [2, 2, 2] * 1),
         ('a', [1, 1, 1] * 1),
         ('b', [2, 2, 2] * 1),
         ('Objective', [0.1] * 1 + [0.2] * 1 + [0.3] * 1)]
    ))
    rs = sherpa.algorithms.RandomSearch()
    gs = SequentialTesting(algorithm=rs,
                           K=4,
                           n=(3, 6, 9),
                           P=0.5)
    assert gs._is_stage_done(results_df, stage=2,
                             num_trials_for_stage=3) == True
    assert gs._is_stage_done(results_df, stage=2,
                             num_trials_for_stage=4) == False


def test_prep_df_for_linreg():
    parameters = [sherpa.Choice('a', [0, 1]),
                  sherpa.Choice('b', [3, 4])]

    configs = [{'a': 0, 'b': 3}, {'a': 0, 'b': 4}]

    results_df = pandas.DataFrame(collections.OrderedDict(
        [('Trial-ID', list(range(1, 9))),
         ('Status', [sherpa.TrialStatus.COMPLETED] * 8),
         ('stage', [1] * 8),
         ('a', [0, 0, 0, 0, 1, 1, 1, 1]),
         ('b', [3, 3, 4, 4, 3, 3, 4, 4]),
         ('Objective', [1., 1.1, 2., 2.1, 5., 5.1, 6., 6.1])]
    ))

    rs = sherpa.algorithms.RandomSearch()
    gs = SequentialTesting(algorithm=rs,
                           K=4,
                           n=(3, 6, 9),
                           P=0.5)

    r = gs._prep_df_for_linreg(parameters, results_df, configs, lower_is_better=True)
    cols = sorted(r.columns.tolist())

    r_expected = results_df.loc[results_df.Objective < 5.]
    r_expected = r_expected.assign(Rank=pandas.Series([1, 1, 2, 2]))
    print(r)
    print(r_expected)

    assert r.loc[:, cols].equals(r_expected.loc[:, cols])


def test_get_best_configs():
    parameters = [sherpa.Choice('a', [0, 1]),
                  sherpa.Choice('b', [3, 4])]

    configs = [{'a': 0, 'b': 3},
               {'a': 0, 'b': 4},
               {'a': 1, 'b': 3},
               {'a': 1, 'b': 4}]

    results_df = pandas.DataFrame(collections.OrderedDict(
        [('Trial-ID', list(range(1, 9))),
         ('Status', [sherpa.TrialStatus.COMPLETED] * 8),
         ('stage', [1] * 8),
         ('a', [0, 0, 0, 0, 1, 1, 1, 1]),
         ('b', [3, 3, 4, 4, 3, 3, 4, 4]),
         ('Objective', [1., 1.1, 1.1, 1.2, 5., 5.1, 6., 6.1])]
    ))

    rs = sherpa.algorithms.RandomSearch()
    gs = SequentialTesting(algorithm=rs,
                           K=4,
                           n=(3, 6, 9),
                           P=0.5)

    best_configs = gs._get_best_configs(parameters, results_df, configs,
                                        lower_is_better=True,
                                        alpha=0.05)
    print(best_configs)
    print(configs[0:2])
    assert best_configs == configs[0:2]


def test_get_best_configs_larger_is_better():
    parameters = [sherpa.Choice('a', [0, 1]),
                  sherpa.Choice('b', [3, 4])]

    configs = [{'a': 0, 'b': 3},
               {'a': 0, 'b': 4},
               {'a': 1, 'b': 3},
               {'a': 1, 'b': 4}]

    results_df = pandas.DataFrame(collections.OrderedDict(
        [('Trial-ID', list(range(1, 9))),
         ('Status', [sherpa.TrialStatus.COMPLETED] * 8),
         ('stage', [1] * 8),
         ('a', [0, 0, 0, 0, 1, 1, 1, 1]),
         ('b', [3, 3, 4, 4, 3, 3, 4, 4]),
         ('Objective', [1., 1.1, 1.1, 1.2, 6., 6.1, 6., 6.1])]
    ))

    rs = sherpa.algorithms.RandomSearch()
    gs = SequentialTesting(algorithm=rs,
                           K=4,
                           n=(3, 6, 9),
                           P=0.5)

    best_configs = gs._get_best_configs(parameters, results_df, configs,
                                        lower_is_better=False,
                                        alpha=0.05)
    print(best_configs)
    print(configs[2:])
    assert best_configs == configs[2:]


def test_get_suggestion():
    for _ in range(10):
        parameters = [sherpa.Continuous('myparam', [0, 1]),
                      sherpa.Discrete('myparam2', [0, 10])]
        rs = sherpa.algorithms.RandomSearch()
        gs = SequentialTesting(algorithm=rs,
                               K=10,
                               n=(3, 6, 9),
                               P=0.5)
        study = sherpa.Study(algorithm=gs,
                             parameters=parameters,
                             lower_is_better=True,
                             disable_dashboard=True)
        seen_configs = []
        last_config = {}
        config_count = 3
        for trial in study:
            print(trial.id, trial.parameters, "{}/{}".format(gs.k, gs.K[gs.t]),
                  "{}/{}".format(gs.t, gs.T))
            if trial.parameters == last_config:
                config_count += 1
                assert config_count <= 3
            elif trial.parameters == "DONE":
                assert gs.K[gs.t] == 1 or gs.t == 3
                break
            else:
                assert config_count == 3
                config_count = 1
                last_config = trial.parameters
                if trial.id <= 30:
                    seen_configs.append(trial.parameters['myparam'])
                else:
                    assert trial.parameters['myparam'] in seen_configs
            study.add_observation(trial,
                                  iteration=1,
                                  objective=trial.parameters[
                                                'myparam'] + numpy.random.normal(
                                      scale=0.01))
            study.finalize(trial)


def test_overall_lower_is_better():
    parameters = [sherpa.Continuous('myparam', [0, 10]),
                  sherpa.Discrete('myparam2', [0, 10])]
    rs = sherpa.algorithms.RandomSearch()
    gs = SequentialTesting(algorithm=rs,
                           K=10,
                           n=(3, 6, 9),
                           P=0.5)
    study = sherpa.Study(algorithm=gs,
                         parameters=parameters,
                         lower_is_better=True,
                         disable_dashboard=True)

    for trial in study:
        print(trial.id, trial.parameters, "{}/{}".format(gs.k, gs.K[gs.t]),
              "{}/{}".format(gs.t, gs.T))

        study.add_observation(trial,
                              iteration=1,
                              objective=trial.parameters[
                                            'myparam'] + numpy.random.normal(
                                  scale=1.))
        study.finalize(trial)

    completed = study.results.query("Status == 'COMPLETED'")
    assert completed.myparam.min() in completed[completed.stage == 2].myparam.unique()


def test_overall_larger_is_better():
    parameters = [sherpa.Continuous('myparam', [0, 10]),
                  sherpa.Discrete('myparam2', [0, 10])]
    rs = sherpa.algorithms.RandomSearch()
    gs = SequentialTesting(algorithm=rs,
                           K=10,
                           n=(3, 6, 9),
                           P=0.5)
    study = sherpa.Study(algorithm=gs,
                         parameters=parameters,
                         lower_is_better=False,
                         disable_dashboard=True)

    for trial in study:
        print(trial.id, trial.parameters, "{}/{}".format(gs.k, gs.K[gs.t]),
              "{}/{}".format(gs.t, gs.T))

        study.add_observation(trial,
                              iteration=1,
                              objective=trial.parameters[
                                            'myparam'] + numpy.random.normal(
                                  scale=1.))
        study.finalize(trial)

    completed = study.results.query("Status == 'COMPLETED'")
    assert completed.myparam.max() in completed[completed.stage == 2].myparam.unique()


def test_get_best_result_lower():
    parameters = [sherpa.Choice('a', [0, 1]),
                  sherpa.Choice('b', [3, 4])]

    results_df = pandas.DataFrame(collections.OrderedDict(
        [('Trial-ID', list(range(1, 9))),
         ('Status', [sherpa.TrialStatus.COMPLETED] * 8),
         ('stage', [1] * 8),
         ('a', [0, 0, 0, 0, 1, 1, 1, 1]),
         ('b', [3, 3, 4, 4, 3, 3, 4, 4]),
         ('Objective', [1., 1.1, 2.1, 2.2, 5., 5.1, 6., 6.1])]
    ))

    rs = sherpa.algorithms.RandomSearch()
    gs = SequentialTesting(algorithm=rs,
                           K=4,
                           n=(3, 6, 9),
                           P=0.5)

    best_config = gs.get_best_result(parameters, results_df,
                                      lower_is_better=True)
    assert best_config == {'a': 0, 'b': 3, 'MeanObjective': 1.05}


def test_get_best_result_larger():
    parameters = [sherpa.Choice('a', [0, 1]),
                  sherpa.Choice('b', [3, 4])]

    results_df = pandas.DataFrame(collections.OrderedDict(
        [('Trial-ID', list(range(1, 9))),
         ('Status', [sherpa.TrialStatus.COMPLETED] * 8),
         ('stage', [1] * 8),
         ('a', [0, 0, 0, 0, 1, 1, 1, 1]),
         ('b', [3, 3, 4, 4, 3, 3, 4, 4]),
         ('Objective', [1., 1.1, 2.1, 2.2, 5., 5.1, 6., 6.1])]
    ))

    rs = sherpa.algorithms.RandomSearch()
    gs = SequentialTesting(algorithm=rs,
                           K=4,
                           n=(3, 6, 9),
                           P=0.5)

    best_config = gs.get_best_result(parameters, results_df,
                                      lower_is_better=False)
    assert best_config == {'a': 1, 'b': 4, 'MeanObjective': 6.05}


# def test_type_I_error():
#     false_positives = 0
#     K = 50
#     nreps = 100
#     for _ in range(nreps):
#         parameters = [sherpa.Continuous('myparam', [0, 1])]
#         rs = sherpa.algorithms.RandomSearch()
#         gs = SequentialTesting(algorithm=rs,
#                                K=K,
#                                n=(3, 6, 9),
#                                P=0.5)
#         study = sherpa.Study(algorithm=gs,
#                              parameters=parameters,
#                              lower_is_better=True,
#                              disable_dashboard=True)
#         seen_configs = []
#         last_config = {}
#         config_count = 3
#         for trial in study:
#             study.add_observation(trial,
#                                   iteration=1,
#                                   objective=numpy.random.normal())
#             study.finalize(trial)
#         if gs.K.get(3, 0) < K:
#             false_positives += 1
#     print(float(false_positives) / nreps)
#     assert float(false_positives) / nreps == 0.05


def test_wait():
    parameters = [sherpa.Continuous('myparam', [0, 1])]
    rs = sherpa.algorithms.RandomSearch()
    gs = SequentialTesting(algorithm=rs,
                           K=10,
                           n=(3, 6, 9),
                           P=0.5)
    study = sherpa.Study(algorithm=gs,
                         parameters=parameters,
                         lower_is_better=True,
                         disable_dashboard=True)

    for _ in range(10*3 - 1):
        trial = study.get_suggestion()
        print(trial.id, trial.parameters, "{}/{}".format(gs.k, gs.K[gs.t]),
              "{}/{}".format(gs.t, gs.T))
        study.add_observation(trial,
                              iteration=1,
                              objective=trial.parameters['myparam'] + numpy.random.normal(
                                  scale=0.01))
        study.finalize(trial)

    trial = study.get_suggestion()
    assert trial.parameters['stage'] == 1

    waittrial = study.get_suggestion()
    assert waittrial == 'WAIT'
    study.add_observation(trial,
                          iteration=1,
                          objective=trial.parameters['myparam'] + numpy.random.normal(
                              scale=0.01))
    study.finalize(trial)

    trial = study.get_suggestion()
    assert trial.parameters['stage'] == 2

testscript = """import sherpa
import time
import numpy

client = sherpa.Client()
trial = client.get_trial()
client.send_metrics(trial=trial, iteration=1,
                    objective=numpy.random.normal())
"""


def test_parallel():
    tempdir = tempfile.mkdtemp(dir=".")

    parameters = [sherpa.Continuous('myparam', [0, 1])]
    rs = sherpa.algorithms.RandomSearch()
    gs = SequentialTesting(algorithm=rs,
                           K=3,
                           n=(3, 6, 9),
                           P=0.5,
                           verbose=True)

    scheduler = sherpa.schedulers.LocalScheduler()

    filename = os.path.join(tempdir, "test.py")
    with open(filename, 'w') as f:
        f.write(testscript)

    try:
        results = sherpa.optimize(parameters=parameters,
                                  algorithm=gs,
                                  lower_is_better=True,
                                  command="python {}".format(filename),
                                  output_dir=tempdir,
                                  scheduler=scheduler,
                                  max_concurrent=2,
                                  verbose=1,
                                  disable_dashboard=True)

    finally:
        shutil.rmtree(tempdir)


def test_results_aggregation():
    parameters = [sherpa.Continuous('myparam', [0, 1])]

    class MyAlg(Algorithm):
        def get_suggestion(self, parameters, results, lower_is_better):
            if results is not None and len(results) > 0:
                print(results)
                assert 'ObjectiveStdErr' in results.columns
                assert (results.loc[:, 'Objective'] == 0.).all()
                exp_std_err = numpy.sqrt(numpy.var([-1,0,1],ddof=1)/(3-1))
                assert (numpy.isclose(results.loc[:, 'ObjectiveStdErr'], exp_std_err)).all()
            return {'myparam': numpy.random.random()}


    alg = MyAlg()
    gs = SequentialTesting(algorithm=alg,
                           K=10,
                           n=(3, 6, 9),
                           P=0.5)
    study = sherpa.Study(algorithm=gs,
                         parameters=parameters,
                         lower_is_better=True,
                         disable_dashboard=True)
    for trial in study:
        study.add_observation(trial,
                              iteration=1,
                              objective=trial.id%3-1)
        study.finalize(trial)
        print(study.results)


def test_bayes_opt():
    def f(x, sd=1):
        y = (x - 3) ** 2 + 10.
        if sd == 0:
            return y
        else:
            return y + numpy.random.normal(loc=0., scale=sd,
                                           size=numpy.array(x).shape)

    parameters = [sherpa.Continuous('x', [1, 6])]

    alg = GPyOpt(max_num_trials=10)
    gs = SequentialTesting(algorithm=alg,
                           K=10,
                           n=(3, 6, 9),
                           P=0.5)
    study = sherpa.Study(algorithm=gs,
                         parameters=parameters,
                         lower_is_better=True,
                         disable_dashboard=True)
    for trial in study:
        study.add_observation(trial,
                              iteration=1,
                              objective=f(trial.parameters['x']))
        study.finalize(trial)
        print(study.results)


if __name__ == '__main__':
    test_parallel()