from __future__ import print_function
from __future__ import absolute_import
import pytest
import tempfile
import numpy as np
import pandas as pd
import math
import shutil
from hobbit.core import Hyperparameter
from hobbit import hparam_generators
from hobbit.resultstable import ResultsTable
from hobbit.utils.testing_utils import branin


def get_example_hparam_ranges():
    """
    These are a set of hyper-parameter ranges for tests below.
    """
    return [Hyperparameter('learning_rate', (0.0001, 0.01), 'log-uniform'),
            Hyperparameter('activation', [['sigmoid', 'tanh', 'relu']], 'choice'),
            Hyperparameter('dropout', {"low": 0., "high": 1.}, 'uniform')]

@pytest.mark.run(order=2)
def test_sample_from():
    """
    Test the sample_from function
    """
    assert 0. <= hparam_generators.sample_from('uniform', (0., 1.)) <= 1.
    assert hparam_generators.sample_from('choice',
                                         [('A', 'B', 'C')]) in ('A', 'B', 'C')

@pytest.mark.run(order=1)
def test_random_generator():
    """
    Get example ranges, create RandomGenerator, check that result
    is within ranges
    """
    hparam_ranges = get_example_hparam_ranges()
    random_gen = hparam_generators.RandomGenerator(hparam_ranges)
    for _ in range(100):
        hparams = random_gen.next()
        assert 0.0001 <= hparams['learning_rate'] <= 0.01
        assert hparams['activation'] in ('sigmoid', 'tanh', 'relu')
        assert 0. <= hparams['dropout'] <= 1.


def test_get_grid():
    df = pd.DataFrame({'param_a': 0., 'param_b': 0.})

    hparam_ranges = [Hyperparameter('param_a', {"low": 0., "high": 1.},
                                    'uniform'),
                     Hyperparameter('param_b', {"low": 0., "high": 1.},
                                    'uniform')
                     ]
    gp_ei = hparam_generators.GaussianProcessEI(hparam_ranges)

    grid = gp_ei.get_grid(df)
    assert grid.shape[0] == 11*11
    for i in np.arange(0., 1., 0.1):
        for j in np.arange(0, 1., 0.1):
            assert np.array([i, j]) in grid

def test_turn_array_to_hparam_dict():
    gp_ei = hparam_generators.GaussianProcessEI(get_example_hparam_ranges())
    d = {'activation_relu': {1: 0, 2: 1, 3: 0},
         'activation_sigmoid': {1: 0, 2: 0, 3: 1},
         'activation_tanh': {1: 1, 2: 0, 3: 0},
         'dropout': {1: 0.67389113354825525,
          2: 0.34886201630561509,
          3: 0.27259982218128875},
         'learning_rate': {1: 0.00012199066883461425,
          2: 0.00012022855306991257,
          3: 0.017118936066349882}}
    df = pd.DataFrame(d)
    hparams = gp_ei.turn_array_to_hparam_dict(X_array=np.array([0.,0.,1.,0.3,
                                                             0.4]), X_df=df)
    assert hparams['activation'] == 'tanh'
    assert np.isclose(hparams['dropout'], 0.3)
    assert np.isclose(hparams['learning_rate'], 0.4)


def test_gaussian_process_expected_improvement_on_parabola():
    """
    Strategy: make two parameters each between 0 and 1,
    let their loss be equal to their product,
    see if the bayesian optimization finds that they should both be 0.

    """
    tmp_folder = tempfile.mkdtemp(prefix='test_repo')

    hparam_ranges = [Hyperparameter('param_a', {"low": 0., "high": 1.},
                                    'uniform'),
                     Hyperparameter('param_b', {"low": 0., "high": 1.},
                                    'uniform')
                     ]
    gp_ei = hparam_generators.GaussianProcessEI(hparam_ranges)

    rt = ResultsTable(tmp_folder)

    get_loss = lambda hparam_dict: np.sum([(hparam_dict[key]-0.2)**2 for key in
                                               hparam_dict])

    for i in range(50):
        hparams = gp_ei.next(X=rt.get_hparams_df(), y=rt.get_column('Loss'))
        loss = get_loss(hparams)
        print(hparams, loss)
        rt.set(run_id=(1, i), epochs=1, loss=loss,
               hparams=hparams)

    assert np.isclose(hparams['param_a'], 0.2) and \
           np.isclose(hparams['param_b'], 0.2)

    shutil.rmtree(tmp_folder)

def test_gaussian_process_expected_improvement_on_braninhoo():
    """
    Strategy: make two parameters each between 0 and 1,
    let their loss be equal to their product,
    see if the bayesian optimization finds that they should both be 0.

    """
    tmp_folder = tempfile.mkdtemp(prefix='test_repo')

    hparam_ranges = [Hyperparameter('param_a', {"low": -5., "high": 10.},
                                    'uniform'),
                     Hyperparameter('param_b', {"low": 0., "high": 15.},
                                    'uniform')
                     ]
    gp_ei = hparam_generators.GaussianProcessEI(hparam_ranges,
                                                num_grid_points=200)

    rt = ResultsTable(tmp_folder)

    get_loss = lambda hparam_dict: branin(hparam_dict['param_a'],
                                          hparam_dict['param_b'])

    for i in range(50):
        hparams = gp_ei.next(X=rt.get_hparams_df(), y=rt.get_column('Loss'))
        loss = get_loss(hparams)
        print(hparams, loss)

        rt.set(run_id=(1, i), epochs=1, loss=loss,
               hparams=hparams)

    print(rt.get_table())

    assert np.isclose(loss, 0.397887) and (np.isclose([hparams['param_a'],
                                                      hparams['param_b']],
                                                     [-np.pi,12.275]) or \
           np.isclose([hparams['param_a'], hparams['param_b']], [np.pi,
                                                                 2.275]) or \
           np.isclose([hparams['param_a'], hparams['param_b']], [9.42478,
                                                                 2.475]))

    shutil.rmtree(tmp_folder)








if __name__ == '__main__':
    # pytest.main([__file__])
    test_gaussian_process_expected_improvement_on_parabola()