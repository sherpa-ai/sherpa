from __future__ import print_function
from __future__ import absolute_import
from sherpa.utils.testing_utils import load_dataset
from sherpa.utils.testing_utils import create_model_two as my_model
import numpy as np


def mnist_demo():
    from sherpa.algorithms import Hyperband
    from sherpa import Hyperparameter

    tmp_folder = './test_repo'

    train_data, valid_data = load_dataset()

    my_hparam_ranges = [Hyperparameter(name='learning_rate', distr_args=(0.0001, 0.1), distribution='log-uniform'),
                        Hyperparameter(name='activation', distr_args=[('sigmoid', 'tanh', 'relu')], distribution='choice'),
                        Hyperparameter(name='dropout', distr_args=(0., 0.8),
                                       distribution='uniform')]


    hband = Hyperband(model_function=my_model,
                        hparam_ranges=my_hparam_ranges,
                        dataset=train_data,
                        repo_dir=tmp_folder,
                        validation_data=valid_data)

    tab = hband.run(R=16, eta=2)

    print(tab)


def mnist_legoband():
    from sherpa.algorithms import Legoband
    from sherpa import GrowingHyperparameter

    tmp_folder = './test_repo'

    train_data, valid_data = load_dataset()

    my_hparam_ranges = [GrowingHyperparameter(name='learning_rate',
                                              choices=[0.0001, 0.001, 0.01,
                                                        0.1],
                                              start_value=10),
                        GrowingHyperparameter(name='activation',
                                              choices=['sigmoid', 'tanh',
                                                        'relu'],
                                              start_value=10),
                        GrowingHyperparameter(name='dropout',
                                              choices=[0.00001, 0.1, 0.2, 0.3,
                                                        0.4, 0.5, 0.6, 0.7],
                                              start_value=10)]


    hband = Legoband(model_function=my_model,
                     hparam_ranges=my_hparam_ranges,
                     dataset=train_data,
                     repo_dir=tmp_folder,
                     validation_data=valid_data)

    tab = hband.run(R=20, eta=3)

    print(tab)


def mnist_natural_selection():
    from sherpa.algorithms import NaturalSelection
    from sherpa import GrowingHyperparameter

    tmp_folder = './ns_test_repo'

    train_data, valid_data = load_dataset()

    my_hparam_ranges = [GrowingHyperparameter(name='learning_rate',
                                              choices=list(10.**np.linspace(
                                                  np.log10(0.0001),
                                                  np.log10(0.1), 8)),
                                              start_value=2.),
                        GrowingHyperparameter(name='activation',
                                              choices=['sigmoid', 'tanh',
                                                        'relu'],
                                              start_value=0.1),
                        GrowingHyperparameter(name='dropout',
                                              choices=list(np.linspace(
                                                  0.001,0.8,8)),
                                              start_value=5.)]


    hband = NaturalSelection(model_function=my_model,
                     hparam_ranges=my_hparam_ranges,
                     dataset=train_data,
                     repo_dir=tmp_folder,
                     validation_data=valid_data)

    tab = hband.run(factor=5)

    print(tab)


if __name__ == '__main__':
    mnist_natural_selection()