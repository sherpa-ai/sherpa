from __future__ import print_function
from __future__ import absolute_import
from sherpa.utils.testing_utils import create_model, load_dataset, store_mnist_hdf5, get_hdf5_generator
import tempfile
import shutil
import os
import pytest


def test_bayesian_optimization():
    """
    # Strategy

    Make a model where we know with certainty which model should be the best based on the hyperparameters, here number
    of units. For each run sort by number of units and then by the number of epochs. Then test that this is a
    non-increasing sequence in the number of units. Note second sort needs to be stable

    """
    from sherpa.algorithms import BayesianOptimization
    from sherpa import Hyperparameter

    tmp_folder = tempfile.mkdtemp(prefix='test_repo')

    train_dataset, valid_dataset = load_dataset(short=False)

    my_hparam_ranges = [Hyperparameter(name='num_units',
                                       distr_args=[1, 50],
                                       distribution='uniform'),
                        Hyperparameter(name='lr', distr_args=[0.01, 0.1],
                                       distribution='uniform')]


    bo = BayesianOptimization(model_function=create_model,
                                dataset=train_dataset,
                                validation_data=valid_dataset,
                                hparam_ranges=my_hparam_ranges,
                                repo_dir=tmp_folder)

    tab = bo.run(num_experiments=15, num_epochs=1)

    print(tab)

    # Create a column for the hparam['num_units']
    # tab['num_units'] = [eval(hparam)['num_units'] for hparam in tab['Hparams']]

    # for run_number in tab['Run'].unique():
    #     run = tab[tab['Run']==run_number]
    #     sorted_run = run.sort_values(by='num_units', ascending=False).sort_values(by='Epochs', ascending=False, kind='quicksort')
    #     l = list(sorted_run['num_units'])
    #     assert all(l[i] >= l[i+1] for i in range(len(l)-1)), "Hyperband did not select expected model to continue" \
    #                                                          " training"
    #
    #
    # # check number of files in repo == 17 + 1 for csv
    # assert len([fname for fname in os.listdir(tmp_folder) if fname.endswith('.csv') or
    #             fname.endswith('.pkl') or fname.endswith('.h5')]) == 17*2 + 1

    shutil.rmtree(tmp_folder)


if __name__=='__main__':
    test_bayesian_optimization()