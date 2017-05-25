from __future__ import print_function
from __future__ import absolute_import
from sherpa.utils.testing_utils import create_model, load_dataset
import tempfile
import shutil
import os
import pytest


def test_legoband():
    """
    # Strategy

    Make a model where we know with certainty which model should be the best based on the hyperparameters, here number
    of units. For each run sort by number of units and then by the number of epochs. Then test that this is a
    non-increasing sequence in the number of units. Note second sort needs to be stable

    """
    from sherpa.algorithms import Legoband
    from sherpa import GrowingHyperparameter

    tmp_folder = tempfile.mkdtemp(prefix='test_repo')

    train_dataset, valid_dataset = load_dataset(short=False)

    my_hparam_ranges = [GrowingHyperparameter(name='num_units', choices=[1,
                                                                         5, 50]),
                        GrowingHyperparameter(name='lr', choices=[0.01,0.05,
                                                                  0.1])]


    lband = Legoband(model_function=create_model,
                                dataset=train_dataset,
                                validation_data=valid_dataset,
                                hparam_ranges=my_hparam_ranges,
                                repo_dir=tmp_folder)

    tab = lband.run(R=20, eta=3)

    print(tab)

    shutil.rmtree(tmp_folder)


def test_natural_selection():
    """
    # Strategy

    Make a model where we know with certainty which model should be the best based on the hyperparameters, here number
    of units. For each run sort by number of units and then by the number of epochs. Then test that this is a
    non-increasing sequence in the number of units. Note second sort needs to be stable

    """
    from sherpa.algorithms import NaturalSelection
    from sherpa import GrowingHyperparameter

    tmp_folder = tempfile.mkdtemp(prefix='test_repo')

    train_dataset, valid_dataset = load_dataset(short=False)

    my_hparam_ranges = [GrowingHyperparameter(name='num_units', choices=[1,
                                                                         5, 50]),
                        GrowingHyperparameter(name='lr', choices=[0.01,0.05,
                                                                  0.1])]


    ns = NaturalSelection(model_function=create_model,
                                dataset=train_dataset,
                                validation_data=valid_dataset,
                                hparam_ranges=my_hparam_ranges,
                                repo_dir=tmp_folder)

    tab = ns.run(factor=4, survivors=2)

    print(tab)

    shutil.rmtree(tmp_folder)

if __name__=='__main__':
    test_natural_selection()