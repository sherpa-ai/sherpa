from __future__ import print_function
from __future__ import absolute_import
from hobbit.utils.testing_utils import load_dataset
from hobbit.utils.testing_utils import create_model
import tempfile
import shutil
import os
import pytest


@pytest.mark.run(order=9)
def test_hyperband():
    """
    # Strategy

    Make a model where we know with certainty which model should be the best based on the hyperparameters, here number
    of units. For each run sort by number of units and then by the number of epochs. Then test that this is a
    non-increasing sequence in the number of units. Note second sort needs to be stable

    """
    from hobbit.algorithms import Hyperband
    from hobbit import Hyperparameter

    tmp_folder = tempfile.mkdtemp(prefix='test_repo')

    my_dataset = load_dataset(short=True)

    my_hparam_ranges = [Hyperparameter(name='num_units', distr_args=[(1, 5, 50)], distribution='choice'),
                        Hyperparameter(name='lr', distr_args=[(0.01,)], distribution='choice')]


    hband = Hyperband(model_function=create_model,
                        dataset=my_dataset,
                        hparam_ranges=my_hparam_ranges,
                        repo_dir=tmp_folder)

    tab = hband.run(R=20, eta=3)

    # Create a column for the hparam['num_units']
    tab['num_units'] = [eval(hparam)['num_units'] for hparam in tab['Hparams']]

    for run_number in tab['Run'].unique():
        run = tab[tab['Run']==run_number]
        sorted_run = run.sort_values(by='num_units', ascending=False).sort_values(by='Epochs', ascending=False, kind='quicksort')
        l = list(sorted_run['num_units'])
        assert all(l[i] >= l[i+1] for i in range(len(l)-1)), "Hyperband did not select expected model to continue" \
                                                             " training"

    # Check all epochs are the expected numbers
    assert all(i in [2, 9, 29, 7, 27, 20] for i in tab['Epochs']), "Unexpected number of epochs for a model"

    # check number of files in repo == 17 + 1 for csv
    assert len([fname for fname in os.listdir(tmp_folder) if fname.endswith('.csv') or
                fname.endswith('.pkl') or fname.endswith('.h5')]) == 17*2 + 1

    shutil.rmtree(tmp_folder)


if __name__ == '__main__':
    pytest.main([__file__])
