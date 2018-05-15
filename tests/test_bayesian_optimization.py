import sherpa
from sherpa.algorithms import BayesianOptimization
import numpy as np
import pandas as pd
import math


def test_transformers():
    parameter = sherpa.Choice('choice', ['a', 'b', 'c', 'd'])
    transformer = BayesianOptimization.ChoiceTransformer(parameter)
    assert np.all(transformer.transform(['d', 'c', 'b', 'a'])
                  == np.flip(np.eye(4), axis=0))

    assert all(transformer.reverse(transformer.transform(['d', 'c', 'b', 'a']))
               == np.array(['d', 'c', 'b', 'a']))

    parameter = sherpa.Continuous('continuous', [0., 0.4])
    transformer = BayesianOptimization.ContinuousTransformer(parameter)
    assert np.all(transformer.transform([0.2, 0.4, 0.]) == np.array([0.5, 1.0, 0.0]))
    assert np.all(transformer.reverse(transformer.transform([0.2, 0.4, 0.]))
                  == np.array([0.2, 0.4, 0.]))

    parameter = sherpa.Continuous('continuous-log', [0.00001, 0.1], 'log')
    transformer = BayesianOptimization.ContinuousTransformer(parameter)
    print(transformer.transform([0.01]))
    assert np.all(transformer.transform([0.0001, 0.001, 0.01]) == np.array(
        [0.25, 0.5, 0.75]))
    print(transformer.reverse(
        transformer.transform([0.0001, 0.001, 0.01])))
    assert np.all(transformer.reverse(
        transformer.transform([0.0001, 0.001, 0.01])) == np.array(
        [0.0001, 0.001, 0.01]))

    parameter = sherpa.Discrete('discrete', [0, 12])
    transformer = BayesianOptimization.DiscreteTransformer(parameter)
    assert np.all(transformer.transform([3, 6, 9])
                  == np.array([0.25, 0.5, 0.75]))
    assert np.all(
        transformer.reverse(transformer.transform([3, 6, 9])) == np.array(
            [3, 6, 9]))
    assert np.all(transformer.reverse([0.2, 0.3, 0.4]) == np.array([2, 4, 5]))

    parameter = sherpa.Discrete('discrete-log', [10, 100000], 'log')
    transformer = BayesianOptimization.DiscreteTransformer(parameter)
    assert np.all(transformer.transform([10, 100, 1000, 10000, 100000])
                  == np.array([0., 0.25, 0.5, 0.75, 1.]))
    assert np.all(transformer.reverse(
        transformer.transform([10, 100, 1000, 10000, 100000])) == np.array(
        [10, 100, 1000, 10000, 100000]))

    # parameter = sherpa.Ordinal('ordinal', ['one', 'two', 'three'])
    # transformer = BayesianOptimization.DiscreteTransformer(parameter)
    # assert np.all(transformer.transform(['one', 'three', 'two'])
    #               == np.array([0., 1., 0.5]))
    # assert np.all(transformer.reverse(transformer.transform(['one', 'three', 'two']))
    #               == np.array(['one', 'three', 'two']))
    # assert np.all(transformer.reverse([0.2, 0.7, 0.9]))== np.array(['one', 'three', 'two']))


def test_design():
    parameters = [sherpa.Choice('choice', ['a', 'b', 'c', 'd']),
                  sherpa.Continuous('continuous', [0., 0.4]),
                  sherpa.Discrete('discrete', [0, 12])]

    bayesian_optimization = BayesianOptimization()
    bayesian_optimization.num_candidates = 100

    candidates = bayesian_optimization._generate_candidates(parameters)
    assert len(candidates) == bayesian_optimization.num_candidates
    assert len(candidates.columns) == len(parameters)

    X = bayesian_optimization._to_design(candidates, parameters)
    assert X.shape == (bayesian_optimization.num_candidates, 6)
    for j in range(X.shape[1]):
        assert (X[:, j] >= 0.).all() and (X[:, j] <= 1.).all()

    df = bayesian_optimization._from_design(X)
    pd.testing.assert_frame_equal(df, candidates)

    row = bayesian_optimization._from_design(X[0])
    row_dict = row.iloc[0].to_dict()
    candidates_dict = candidates.iloc[0].to_dict()
    assert row_dict['choice'] == candidates_dict['choice']
    assert row_dict['discrete'] == candidates_dict['discrete']
    assert np.isclose(row_dict['continuous'], candidates_dict['continuous'])


def test_optimize():
    # Test Continuous
    parameters = [sherpa.Continuous('continuous', [0., 1,])]

    bayesian_optimization = BayesianOptimization()
    bayesian_optimization.num_candidates = 100

    candidates = bayesian_optimization._generate_candidates(parameters)
    X = bayesian_optimization._to_design(candidates, parameters)

    fun = lambda x: -1*(x-0.5)**2

    Xoptimized, fun_values = bayesian_optimization._maximize(X, fun)
    assert np.isclose(Xoptimized[fun_values.argmax()][0], 0.5)

    # Test Discrete
    parameters = [sherpa.Discrete('discrete', [0, 100])]

    bayesian_optimization = BayesianOptimization()
    bayesian_optimization.num_candidates = 100

    candidates = bayesian_optimization._generate_candidates(parameters)
    X = bayesian_optimization._to_design(candidates, parameters)

    fun = lambda x: -1.*(x-0.5)**2
    Xoptimized, fun_values = bayesian_optimization._maximize(X, fun)
    assert np.isclose(Xoptimized[fun_values.argmax()][0], 0.5)


def test_optimize_mix():
    # Test a mixture of these
    parameters = [sherpa.Continuous('continuous', [0., 1,]),
                  sherpa.Choice('choice', [1, 2, 3, 4]),
                  sherpa.Choice('choice2', [1, 2, 3]),
                  sherpa.Discrete('discrete', [0, 100])]

    # Function should have maximum: [0.5, 0, 0, 0, 1, 0, 0, 1, 0.5]
    # Maximum should be 7
    def fun(x):
        cont = -1.*(x[0]-0.5)**2
        ch = np.dot(x[1:5], np.array([1, 2, 3, 4]))
        ch2 = np.dot(x[5:8], np.array([1, 2, 3]))
        discr = -1.*(x[-1]-0.5)**2
        return cont + ch + ch2 + discr

    bayesian_optimization = BayesianOptimization()
    bayesian_optimization.num_candidates = 100

    candidates = bayesian_optimization._generate_candidates(parameters)
    X = bayesian_optimization._to_design(candidates, parameters)

    Xoptimized, fun_values = bayesian_optimization._maximize(X, fun)
    # print(Xoptimized)
    # print(fun_values)
    print(Xoptimized[fun_values.argmax()])
    print(fun_values.max())
    assert np.all(np.isclose(Xoptimized[fun_values.argmax()],
                             np.array([0.5, 0., 0., 0., 1., 0., 0., 1., 0.5])))


def test_strip_add_choice():
    parameters = [sherpa.Choice('choice', ['a', 'b', 'c', 'd']),
                  sherpa.Continuous('continuous', [0., 0.4]),
                  sherpa.Choice('choice2', [1, 2, 3]),
                  sherpa.Discrete('discrete', [0, 12])]

    bayesian_optimization = BayesianOptimization()
    bayesian_optimization.num_candidates = 5

    candidates = bayesian_optimization._generate_candidates(parameters)

    X = bayesian_optimization._to_design(candidates, parameters)
    for i, row in enumerate(X):
        print(row)
        x, args = bayesian_optimization._strip_choice(row)
        # print("x: ", x)
        # print("args: ", args)
        new_row = bayesian_optimization._add_choice(x, *args)
        print(new_row)
        assert np.all(row == new_row)


def test_1d():
    def obj_func(x):
        # Global maximum of 4 is at x=4
        return 4. * np.exp(-(x - 4.) ** 2 / 10.) * np.cos(1.5 * (x - 4.)) ** 2

    parameters = [sherpa.Continuous('x1', [0., 7.])]

    bayesian_optimization = BayesianOptimization(num_grid_points=5,
                                                 max_num_trials=50,
                                                 fine_tune=False)
    study = sherpa.Study(algorithm=bayesian_optimization,
                         parameters=parameters,
                         lower_is_better=False,
                         disable_dashboard=True)

    for trial in study:
        print("Trial {}:\t{}".format(trial.id, trial.parameters))

        fval = obj_func(trial.parameters['x1'])
        print("Function Value: {}".format(fval))
        study.add_observation(trial=trial,
                              iteration=1,
                              objective=fval)
        study.finalize(trial, status='COMPLETED')
    rval = study.get_best_result()
    assert np.isclose(rval['Objective'], 4.)


def test_convex():
    def convex(x1, x2):
        # Global minimum is at x1=3., x2=5.
        return (x1-3.)**2 + (x2-5.)**2 + 0.1

    parameters = [sherpa.Continuous('x1', [-5., 10.]),
                  sherpa.Continuous('x2', [0., 15.])]

    bayesian_optimization = BayesianOptimization(num_grid_points=2,
                                                 max_num_trials=50,
                                                 fine_tune=True)
    study = sherpa.Study(algorithm=bayesian_optimization,
                         parameters=parameters,
                         lower_is_better=True,
                         disable_dashboard=True)

    for trial in study:
        print("Trial {}:\t{}".format(trial.id, trial.parameters))

        fval = convex(trial.parameters['x1'], trial.parameters['x2'])
        print("Function Value: {}".format(fval))
        study.add_observation(trial=trial,
                              iteration=1,
                              objective=fval)
        study.finalize(trial, status='COMPLETED')

    rval = study.get_best_result()
    print(rval)
    assert np.isclose(rval['Objective'], 0.1, rtol=1e-3)


def test_branin():
    def branin(x1, x2):
        # Global minimum 0.397887 at (-pi, 12.275), (pi, 2.275),
        # and (9.42478, 2.475)
        a = 1
        b = 5.1/(4*math.pi**2)
        c = 5/math.pi
        r = 6
        s = 10
        t = 1/(8*math.pi)
        return a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*math.cos(x1)+s

    parameters = [sherpa.Continuous('x1', [-5., 10.]),
                  sherpa.Continuous('x2', [0., 15.])]

    bayesian_optimization = BayesianOptimization(num_grid_points=2, max_num_trials=50, fine_tune=True)
    study = sherpa.Study(algorithm=bayesian_optimization,
                         parameters=parameters,
                         lower_is_better=True,
                         disable_dashboard=True)

    for trial in study:
        print("Trial {}:\t{}".format(trial.id, trial.parameters))

        fval = branin(trial.parameters['x1'], trial.parameters['x2'])
        print("Branin-Hoo: {}".format(fval))
        study.add_observation(trial=trial,
                              iteration=1,
                              objective=fval)
        study.finalize(trial, status='COMPLETED')
    rval = study.get_best_result()
    print(study.get_best_result())
    assert np.isclose(rval['Objective'], 0.397887, rtol=1e-3)


if __name__ == '__main__':
    test_branin()