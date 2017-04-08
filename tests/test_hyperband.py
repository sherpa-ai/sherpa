from __future__ import print_function
from __future__ import absolute_import
from hobbit.utils.testing_utils import load_dataset
from hobbit.utils.testing_utils import create_model_two as my_model
import tempfile
import shutil
import os
import pytest
import argparse

# parser = argparse.ArgumentParser(description='Testing Hyperband.')
# parser.add_argument('-short', help='Option to shorten test by cutting dataset size')
#
# args = parser.parse_args()
# print(args.accumulate(args.integers))

@pytest.mark.run(order=9)
def test_hyperband():
    from hobbit.algorithms import Hyperband
    from hobbit import Hyperparameter

    # tmp_folder = tempfile.mkdtemp(prefix='test_repo')

    tmp_folder = './test_repo'
    assert not os.path.exists(tmp_folder)

    my_dataset = load_dataset(short=False)

    my_hparam_ranges = [Hyperparameter(name='learning_rate', distr_args=(0.0001, 0.1), distribution='log-uniform'),
                        Hyperparameter(name='activation', distr_args=[('sigmoid', 'tanh', 'relu')], distribution='choice'),
                        Hyperparameter(name='dropout', distr_args=(0., 1.), distribution='uniform')]


    hband = Hyperband(model_function=my_model,
                        dataset=my_dataset,
                        hparam_ranges=my_hparam_ranges,
                        repo_dir=tmp_folder)

    assert os.path.exists(tmp_folder)

    tab = hband.run(R=20, eta=3)

    print(tab)

    # check number of files in repo == 17 + 1 for csv
    assert len([fname for fname in os.listdir(tmp_folder) if fname.endswith('.csv') or
                fname.endswith('.pkl') or fname.endswith('.h5')]) == 17*2 + 1

    # shutil.rmtree(tmp_folder)



if __name__ == '__main__':
    # pytest.main([__file__])

    test_hyperband()
