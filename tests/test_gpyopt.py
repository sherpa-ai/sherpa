import sherpa
import pandas
import pytest
import numpy
import collections
from sherpa.algorithms.bayesian_optimization import GPyOpt


@pytest.fixture
def parameters():
    parameters = [sherpa.Continuous('dropout', [0., 0.5]),
                  sherpa.Continuous('lr', [1e-7, 1e-1], 'log'),
                  sherpa.Choice('activation', ['relu', 'tanh', 'sigmoid']),
                  sherpa.Discrete('num_hidden', [100, 300]),
                  sherpa.Discrete('batch_size', [1, 1000], 'log')
                  ]
    return parameters

@pytest.fixture
def transforms(parameters):
    return GPyOpt._initialize_transforms(parameters)

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
             ('batch_size', [10, 100, 1000]),
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
             ('batch_size', list(10**numpy.random.uniform(1, 3, 10).astype('int'))),
             ('Objective', list(numpy.random.random(10)))]
        ))


# Test setup
def test_infer_num_initial_data_points(parameters):
    assert GPyOpt._infer_num_initial_data_points('infer', parameters) == 6
    assert GPyOpt._infer_num_initial_data_points(3, parameters) == 6
    assert GPyOpt._infer_num_initial_data_points(7, parameters) == 7


def test_processing_of_initial_data_points(parameters):

    data_points_df = pandas.DataFrame(collections.OrderedDict(
                                    [('dropout', [0.5, 0.1, 0.3]),
                                     ('lr', [1e-4, 1e-3, 1e-7]),
                                     ('activation', ['relu', 'tanh', 'sigmoid']),
                                     ('num_hidden', [101, 202, 299]),
                                     ('batch_size', [1, 10, 100])]
                                ))

    data_points = GPyOpt._process_initial_data_points(data_points_df,
                                                      parameters)

    assert data_points == [{'activation': 'relu', 'dropout': 0.5, 'lr': 0.0001,
                            'num_hidden': 101, 'batch_size': 1},
                           {'activation': 'tanh', 'dropout': 0.1, 'lr': 0.001,
                            'num_hidden': 202, 'batch_size': 10},
                           {'activation': 'sigmoid', 'dropout': 0.3,
                            'lr': 1e-07, 'num_hidden': 299, 'batch_size': 100}]

    with pytest.raises(ValueError):
        data_points_missing = GPyOpt._process_initial_data_points(
            data_points_df.drop("activation", axis=1), parameters)


def test_transformations(transforms):
    # transforms
    assert list(map(transforms['dropout'].transform, [0., 0.1, 0.5])) == [0.,
                                                                          0.1,
                                                                          0.5]
    assert list(map(transforms['lr'].transform, [1e-2, 3.3e-4, 0.0001])) == [-2,
                                                                             -3.4814860601221125,
                                                                             -4]
    assert list(map(transforms['activation'].transform,
                    ['relu', 'sigmoid', 'tanh'])) == [0, 2, 1]
    assert list(map(transforms['num_hidden'].transform,
                    [101, 202, 299])) == [101, 202, 299]
    assert list(map(transforms['batch_size'].transform,
                    [1, 10, 100, 1000])) == [0, 1, 2, 3]

    # reverse transforms
    assert list(map(transforms['dropout'].reverse, [0., 0.1, 0.5])) == [0.,
                                                                          0.1,
                                                                          0.5]
    assert list(map(transforms['lr'].reverse, [-2, -3.48, -4])) == [1e-2, 0.0003311311214825911, 0.0001]
    assert list(map(transforms['activation'].reverse,
                    [0, 2, 1])) == ['relu', 'sigmoid', 'tanh']
    assert list(map(transforms['num_hidden'].reverse,
                    [101.1, 201.6, 299.45])) == [101, 202, 299]
    assert list(map(transforms['batch_size'].reverse,
                    [0.001, 1., 2.001, 2.9999])) == [1, 10, 100, 1000]


def test_domain(parameters, transforms):
    domain = GPyOpt._initialize_domain(parameters, transforms)

    assert {'name': 'dropout', 'type': 'continuous',
            'domain': (0., 0.5)} in domain
    assert {'name': 'lr', 'type': 'continuous',
            'domain': (-7, -1)} in domain
    assert {'name': 'activation', 'type': 'discrete',
            'domain': (0, 1, 2)} in domain
    assert {'name': 'num_hidden', 'type': 'continuous',
            'domain': (100, 300)} in domain
    assert {'name': 'batch_size', 'type': 'continuous',
            'domain': (0, 3)} in domain


def test_prepare_data_for_bayes_opt(parameters, results, transforms):
    X, y = GPyOpt._prepare_data_for_bayes_opt(parameters, results, transforms)
    assert numpy.array_equal(X, numpy.array([[0.1, -3., 1, 111, 1],
                                             [0.4, -5., 0, 222, 2],
                                             [0.33, -2., 2, 288, 3]]))

    assert numpy.array_equal(y, numpy.array([[0.1], [0.055], [0.15]]))


def test_reverse_format(parameters, results, transforms):
    X, y = GPyOpt._prepare_data_for_bayes_opt(parameters, results, transforms)

    reversed_X = GPyOpt._reverse_to_sherpa_format(X, transforms, parameters)

    assert reversed_X[0] == {'dropout': 0.1, 'lr': 1e-3, 'activation': 'tanh',
                             'num_hidden': 111, 'batch_size': 10}
    assert reversed_X[1] == {'dropout': 0.4, 'lr': 1e-5, 'activation': 'relu',
                             'num_hidden': 222, 'batch_size': 100}
    assert reversed_X[2] == {'dropout': 0.33, 'lr': 1e-2, 'activation': 'sigmoid',
                             'num_hidden': 288, 'batch_size': 1000}

def test_bayesopt_batch(parameters, results, transforms):
    gpyopt = GPyOpt(max_concurrent=10)
    domain = gpyopt._initialize_domain(parameters, transforms)
    X, y = GPyOpt._prepare_data_for_bayes_opt(parameters, results, transforms)
    batch = gpyopt._generate_bayesopt_batch(domain, X, y, lower_is_better=True)

    assert batch.shape == (10, 5)


def test_overall():
    gpyopt = GPyOpt(max_concurrent=1)
    parameters, results, lower_is_better = sherpa.algorithms.get_sample_results_and_params()

    for i in range(51):
        suggestion = gpyopt.get_suggestion(parameters,
                                           results.loc[results['Trial-ID'] < i,
                                           :],
                                           lower_is_better)
        print(suggestion)

def test_1d():
    def obj_func(x):
        # Global maximum of 4 is at x=4
        return 4. * numpy.exp(-(x - 4.) ** 2 / 10.) * numpy.cos(
            1.5 * (x - 4.)) ** 2

    parameters = [sherpa.Continuous('x1', [0., 7.])]

    bayesian_optimization = GPyOpt(max_concurrent=1,
                                   max_num_trials=50,
                                   model_type='GP',
                                   acquisition_type='EI')
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
    assert numpy.isclose(rval['Objective'], 4., atol=0.2)

if __name__ == '__main__':
    test_1d()