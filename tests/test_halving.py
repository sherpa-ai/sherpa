from __future__ import print_function
from __future__ import absolute_import
from sherpa.utils.testing_utils import create_model, load_dataset, store_mnist_hdf5, get_hdf5_generator
import tempfile
import shutil
import os
import pytest


@pytest.mark.run(order=9)
def test_halving():
    """
    # Strategy

    Make a model where we know with certainty which model should be the best based on the hyperparameters, here number
    of units. For each run sort by number of units and then by the number of epochs. Then test that this is a
    non-increasing sequence in the number of units. Note second sort needs to be stable

    """
    from sherpa.algorithms import Halving
    from sherpa import Hyperparameter

    tmp_folder = tempfile.mkdtemp(prefix='test_repo')

    train_dataset, valid_dataset = load_dataset(short=False)

    my_hparam_ranges = [Hyperparameter(name='num_units', distr_args=[(1, 5, 50)], distribution='choice'),
                        Hyperparameter(name='lr', distr_args=[(0.01,)], distribution='choice')]


    halving = Halving(model_function=create_model,
                        dataset=train_dataset,
                        validation_data=valid_dataset,
                        hparam_ranges=my_hparam_ranges,
                        repo_dir=tmp_folder)

    tab = halving.run(num_models=20, num_start_epochs=1, cut_factor=2)

    # This will train 20-1, 10-2, 5-4, 2-8, 1-16
    # So cumulative epochs will be 1, 3, 7, 15, 31

    # Check all epochs are the expected numbers
    assert all(i in [1, 3, 7, 15, 31] for i in tab['Epochs']), "Unexpected number of epochs for a model"

    # check number of files in repo == 17 + 1 for csv
    assert len([fname for fname in os.listdir(tmp_folder) if fname.endswith('.csv') or
                fname.endswith('.pkl') or fname.endswith('.h5')]) == 20*2 + 1

    shutil.rmtree(tmp_folder)