import sherpa
import pandas
import pytest
import numpy
import collections
import GPyOpt as gpyopt_package
from sherpa.algorithms.bayesian_optimization import GPyOpt
from sherpa.algorithms import Repeat



@pytest.fixture
def parameters():
    parameters = [sherpa.Continuous('dropout', [0., 0.5]),
                  sherpa.Continuous('lr', [1e-7, 1e-1], 'log'),
                  sherpa.Choice('activation', ['relu', 'tanh', 'sigmoid']),
                  sherpa.Discrete('num_hidden', [100, 300])
                  ]
    return parameters

@pytest.fixture
def results():
    return pandas.DataFrame(collections.OrderedDict(
            [('Trial-ID', [1, 2, 3]),
             ('Status', ['COMPLETED']*3),
             ('Iteration', [1]*3),
             ('dropout', [0.1, 0.4, 0.33]),
             ('lr', [1e-3, 1e-5, 1e-2]),
             ('activation', ['tanh', 'relu', 'sigmoid']),
             ('num_hidden', [111, 222, 288]),
             ('Objective', [0.1, 0.055, 0.15])]
        ))

@pytest.fixture
def results_long():
    return pandas.DataFrame(collections.OrderedDict(
            [('Trial-ID', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
             ('Status', ['COMPLETED']*10),
             ('Iteration', [1]*10),
             ('dropout', list(numpy.random.uniform(0, 0.5, 10))),
             ('lr', list(10**numpy.random.uniform(-7, -1, 10))),
             ('activation', list(numpy.random.choice(['relu', 'tanh', 'sigmoid'], 10))),
             ('num_hidden', list(numpy.random.random_integers(100, 300, 10))),
             ('Objective', list(numpy.random.random(10)))]
        ))



@pytest.mark.parametrize("num_initial_data_points,expected",
                     [('infer', 5),
                      (7, 7)])
def test_infer_num_initial_data_points(num_initial_data_points, expected, parameters):
    assert GPyOpt._infer_num_initial_data_points(num_initial_data_points, parameters) == expected


def test_infer_num_initial_data_points_too_few_specified(parameters):
    with pytest.warns(UserWarning, match="num_initial_data_points < number of parameters found"):
        assert GPyOpt._infer_num_initial_data_points(3, parameters) == 5


def test_processing_of_initial_data_points(parameters):

    data_points_df = pandas.DataFrame(collections.OrderedDict(
                                    [('dropout', [0.5, 0.1, 0.3]),
                                     ('lr', [1e-4, 1e-3, 1e-7]),
                                     ('activation', ['relu', 'tanh', 'sigmoid']),
                                     ('num_hidden', [101, 202, 299])]
                                ))

    data_points = GPyOpt._process_initial_data_points(data_points_df,
                                                      parameters)

    assert data_points == [{'activation': 'relu', 'dropout': 0.5, 'lr': 0.0001,
                            'num_hidden': 101},
                           {'activation': 'tanh', 'dropout': 0.1, 'lr': 0.001,
                            'num_hidden': 202},
                           {'activation': 'sigmoid', 'dropout': 0.3,
                            'lr': 1e-07, 'num_hidden': 299}]

    with pytest.raises(ValueError):
        data_points_missing = GPyOpt._process_initial_data_points(
            data_points_df.drop("activation", axis=1), parameters)


@pytest.mark.parametrize("parameters", [([sherpa.Continuous('a', [0, 1])]),
                                    ([sherpa.Continuous('a', [0, 1]), sherpa.Continuous('b', [10., 100])])])
def test_transformation_to_gpyopt_domain_continuous(parameters):
    domain = GPyOpt._initialize_domain(parameters)
    for p, d in zip(parameters, domain):
        assert d['name'] == p.name
        assert d['type'] == 'continuous'
        assert d['domain'] == tuple(p.range)


@pytest.mark.parametrize("parameters", [([sherpa.Continuous('a', [0.1, 1], 'log')]),
                                    ([sherpa.Continuous('a', [0.1, 1], 'log'), sherpa.Continuous('b', [10., 100], 'log')])])
def test_transformation_to_gpyopt_domain_log_continuous(parameters):
    domain = GPyOpt._initialize_domain(parameters)
    for p, d in zip(parameters, domain):
        assert d['name'] == p.name
        assert d['type'] == 'continuous'
        assert d['domain'] == tuple([numpy.log10(p.range[0]),
                                     numpy.log10(p.range[1])])


@pytest.mark.parametrize("parameters", [([sherpa.Discrete('a', [0, 100])]),
                                    ([sherpa.Discrete('a', [1, 100]), sherpa.Discrete('b', [0, 100])])])
def test_transformation_to_gpyopt_domain_discrete(parameters):
    domain = GPyOpt._initialize_domain(parameters)
    for p, d in zip(parameters, domain):
        assert d['name'] == p.name
        assert d['type'] == 'discrete'
        assert d['domain'] == tuple(range(p.range[0], p.range[1]+1))


def test_transformation_to_gpyopt_domain_log_discrete():
    parameters = [sherpa.Discrete('a', [1, 100], 'log')]
    with pytest.warns(UserWarning, match='does not support log-scale'):
        GPyOpt._initialize_domain(parameters)


def test_transformation_to_gpyopt_domain_with_multiple_parameters(parameters):
    domain = GPyOpt._initialize_domain(parameters)

    assert {'name': 'dropout', 'type': 'continuous',
            'domain': (0., 0.5)} == domain[0]
    assert {'name': 'lr', 'type': 'continuous',
            'domain': (-7, -1)} == domain[1]
    assert domain[2]['name'] == 'activation'
    assert domain[2]['type'] == 'categorical'
    assert numpy.array_equal(domain[2]['domain'], numpy.array([0, 1, 2]))
    assert {'name': 'num_hidden', 'type': 'discrete',
            'domain': tuple(range(100,301))} == domain[3]


def test_prepare_data_for_bayes_opt(parameters, results):
    X, y, y_var = GPyOpt._prepare_data_for_bayes_opt(
        parameters, results)
    assert numpy.array_equal(X, numpy.array([[0.1, -3., 1, 111],
                                             [0.4, -5., 0, 222],
                                             [0.33, -2., 2, 288]]))

    assert numpy.array_equal(y, numpy.array([[0.1], [0.055], [0.15]]))


def test_reverse_format(parameters, results):
    X, y, y_var = GPyOpt._prepare_data_for_bayes_opt(parameters,
                                                     results)
    reversed_X = GPyOpt._reverse_to_sherpa_format(X, parameters)

    assert reversed_X[0] == {'dropout': 0.1, 'lr': 1e-3, 'activation': 'tanh',
                             'num_hidden': 111}
    assert reversed_X[1] == {'dropout': 0.4, 'lr': 1e-5, 'activation': 'relu',
                             'num_hidden': 222}
    assert reversed_X[2] == {'dropout': 0.33, 'lr': 1e-2, 'activation': 'sigmoid',
                             'num_hidden': 288}


def test_bayesopt_batch(parameters, results):
    gpyopt = GPyOpt(max_concurrent=10)
    gpyopt.domain = gpyopt._initialize_domain(parameters)
    gpyopt.lower_is_better = True
    X, y, y_var = GPyOpt._prepare_data_for_bayes_opt(parameters, results)
    domain = gpyopt._initialize_domain(parameters)
    batch = gpyopt._generate_bayesopt_batch(X,
                                            y,
                                            lower_is_better=True,
                                            domain=domain)

    assert batch.shape == (10, 4)


def test_types_are_correct(parameters, results):
    gpyopt = GPyOpt(max_concurrent=1)
    suggestion = gpyopt.get_suggestion(parameters,
                                       results,
                                       True)
    assert isinstance(suggestion['dropout'], float)
    assert isinstance(suggestion['lr'], float)
    assert isinstance(suggestion['num_hidden'], int)
    assert isinstance(suggestion['activation'], str)

@pytest.mark.parametrize('lower_is_better,expected_best', [(True, 0), (False, 1)])
def test_get_best_pred(lower_is_better,expected_best):
    results = pandas.DataFrame({'x': numpy.linspace(0, 1, 10),
                                'Objective': numpy.linspace(0, 1, 10),
                                'Status': ['COMPLETED']*10})
    params = [sherpa.Continuous('x', [0, 1])]
    algorithm = GPyOpt(num_initial_data_points=2)
    algorithm.get_suggestion(results=results, parameters=params, lower_is_better=lower_is_better)
    best_params = algorithm.get_best_pred(results=results,
                                          parameters=params,
                                          lower_is_better=lower_is_better)
    assert best_params['x'] == expected_best


@pytest.mark.skip(reason="sample results do not copy when doing `pip install .`")
def test_overall():
    gpyopt = GPyOpt(max_concurrent=1)
    parameters, results, lower_is_better = sherpa.algorithms.get_sample_results_and_params()

    for i in range(51):
        suggestion = gpyopt.get_suggestion(parameters,
                                           results.loc[results['Trial-ID'] < i,
                                           :],
                                           lower_is_better)
        print(suggestion)


@pytest.mark.skip(reason="This test needs to be made deterministic.")
def test_1d_maximize():
    def obj_func(x):
        # Global maximum of 4 is at x=4
        return 4. * numpy.exp(-(x - 4.) ** 2 / 10.) * numpy.cos(
            1.5 * (x - 4.)) ** 2

    parameters = [sherpa.Continuous('x1', [0., 7.])]

    bayesian_optimization = GPyOpt(max_concurrent=1,
                                   max_num_trials=12,
                                   model_type='GP',
                                   acquisition_type='EI',
                                   initial_data_points=[{'x1': 2}, {'x1': 5}],
                                   num_initial_data_points=2)

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
    print(rval)

    # bounds = [{'name': 'x', 'type': 'continuous', 'domain': (0, 7)}]
    # Xinit = numpy.array([2, 5]).reshape(-1, 1)
    # yinit = obj_func(Xinit)
    # myBopt = gpyopt_package.methods.BayesianOptimization(f=obj_func,
    #                                              # function to optimize
    #                                              domain=bounds,
    #                                              # box-constraints of the problem
    #                                              acquisition_type='EI',
    #                                              X=Xinit,
    #                                              y=yinit,
    #                                              initial_design_numdata=0,
    #                                              initial_design_type='random',
    #                                              evaluator_type='local_penalization',
    #                                              batch_size=1,
    #                                              maximize=True,
    #                                              exact_feval=False)
    # # Run the optimization
    # max_iter = 10  # evaluation budget
    # max_time = 60  # time budget
    # eps = 10e-6  # Minimum allows distance between the las two observations
    #
    # myBopt.run_optimization(max_iter, max_time, eps)
    # print(myBopt.get_evaluations())

    assert numpy.isclose(rval['x1'], 4., atol=0.1)


@pytest.mark.skip(reason="This test needs to be made deterministic.")
def test_1d_minimize():
    def obj_func(x):
        # Global maximum of 4 is at x=4
        return -4. * numpy.exp(-(x - 4.) ** 2 / 10.) * numpy.cos(
            1.5 * (x - 4.)) ** 2

    parameters = [sherpa.Continuous('x1', [0., 7.])]

    bayesian_optimization = GPyOpt(max_concurrent=1,
                                   max_num_trials=12,
                                   model_type='GP',
                                   acquisition_type='EI',
                                   initial_data_points=[{'x1': 2}, {'x1': 5}],
                                   num_initial_data_points=2)

    study = sherpa.Study(algorithm=bayesian_optimization,
                         parameters=parameters,
                         lower_is_better=True,
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
    print(rval)

    # bounds = [{'name': 'x', 'type': 'continuous', 'domain': (0, 7)}]
    # Xinit = numpy.array([2, 5]).reshape(-1, 1)
    # yinit = obj_func(Xinit)
    # myBopt = gpyopt_package.methods.BayesianOptimization(f=obj_func,
    #                                              # function to optimize
    #                                              domain=bounds,
    #                                              # box-constraints of the problem
    #                                              acquisition_type='EI',
    #                                              X=Xinit,
    #                                              y=yinit,
    #                                              initial_design_numdata=0,
    #                                              initial_design_type='random',
    #                                              evaluator_type='local_penalization',
    #                                              batch_size=1,
    #                                              maximize=True,
    #                                              exact_feval=False)
    # # Run the optimization
    # max_iter = 10  # evaluation budget
    # max_time = 60  # time budget
    # eps = 10e-6  # Minimum allows distance between the las two observations
    #
    # myBopt.run_optimization(max_iter, max_time, eps)
    # print(myBopt.get_evaluations())

    assert numpy.isclose(rval['x1'], 4., atol=0.1)


@pytest.mark.skip(reason="This test needs to be made deterministic.")
def test_3d():
    def obj_func(x, y, z):
        assert isinstance(x, float)
        assert isinstance(y, str)
        assert isinstance(z, int)
        # Global maximum of 4 is at x=4
        return -4. * numpy.exp(-(x - 4.) ** 2 / 10.) * numpy.cos(
            1.5 * (x - 4.)) ** 2 -int(y) * z

    parameters = [sherpa.Continuous('x', [0., 7.]),
                  sherpa.Choice('y', ["-1", "0", "1"]),
                  sherpa.Discrete('z', [1, 5])]

    bayesian_optimization = GPyOpt(max_concurrent=1,
                                   max_num_trials=100,
                                   model_type='GP',
                                   acquisition_type='EI')

    study = sherpa.Study(algorithm=bayesian_optimization,
                         parameters=parameters,
                         lower_is_better=True,
                         disable_dashboard=True)

    for trial in study:
        print("Trial {}:\t{}".format(trial.id, trial.parameters))

        fval = obj_func(**trial.parameters)
        print("Function Value: {}".format(fval))
        study.add_observation(trial=trial,
                              iteration=1,
                              objective=fval)
        study.finalize(trial, status='COMPLETED')
    rval = study.get_best_result()
    print(rval)

    assert numpy.isclose(rval['x'], 4., atol=0.1)
    assert rval['y'] == 1
    assert rval['z'] == 5


def test_noisy_parabola():
    def f(x, sd=1):
        y = (x - 3) ** 2 + 10.
        if sd == 0:
            return y
        else:
            return y + numpy.random.normal(loc=0., scale=sd,
                                           size=numpy.array(x).shape)

    parameters = [sherpa.Continuous('x1', [0., 7.])]

    bayesian_optimization = GPyOpt(max_concurrent=1,
                                   max_num_trials=5,
                                   model_type='GP',
                                   acquisition_type='EI')
    rep = Repeat(algorithm=bayesian_optimization,
                 num_times=3,
                 agg=True)
    study = sherpa.Study(algorithm=rep,
                         parameters=parameters,
                         lower_is_better=True,
                         disable_dashboard=True)

    for trial in study:
        # print("Trial {}:\t{}".format(trial.id, trial.parameters))

        fval = f(trial.parameters['x1'], sd=1)
        # print("Function Value: {}".format(fval))
        study.add_observation(trial=trial,
                              iteration=1,
                              objective=fval)
        study.finalize(trial, status='COMPLETED')
    # rval = study.get_best_result()
    # print(rval)
    print(study.results.query("Status=='COMPLETED'"))
    # assert numpy.sqrt((rval['Objective'] - 3.)**2) < 0.2


def test_mixed_dtype():
    algorithm = GPyOpt(max_num_trials=4)
    parameters = [
        sherpa.Choice('param_int', [0, 1]),
        sherpa.Choice('param_float', [0.1, 1.1]),
    ]
    study = sherpa.Study(
        parameters=parameters,
        algorithm=algorithm,
        lower_is_better=True,
        disable_dashboard=True,
    )
    for trial in study:
        study.add_observation(trial, iteration=0, objective=0)
        study.finalize(trial)
    assert type(trial.parameters['param_int']) == int
    assert type(trial.parameters['param_float']) == float