from __future__ import print_function
from __future__ import absolute_import
from hobbit.utils.testing_utils import load_dataset
from hobbit.utils.testing_utils import create_model_two as my_model
import tempfile
import shutil
import os
import pytest

@pytest.mark.run(order=8)
def test_random_search():
    from hobbit.algorithms import RandomSearch
    from hobbit import Hyperparameter

    num_experiments = 2
    num_epochs = 2

    tmp_folder = tempfile.mkdtemp(prefix='test_repo')

    train_dataset, valid_dataset = load_dataset()

    my_hparam_ranges = [Hyperparameter('learning_rate', distr_args=(0.0001, 0.1), distribution='log-uniform'),
                        Hyperparameter('activation', distr_args=[('sigmoid', 'tanh', 'relu')], distribution='choice'),
                        Hyperparameter('dropout', distr_args=(0., 1.), distribution='uniform')]

    rs = RandomSearch(model_function=my_model,
                        dataset=train_dataset,
                        validation_data=valid_dataset,
                        hparam_ranges=my_hparam_ranges,
                        repo_dir=tmp_folder)

    tab = rs.run(num_experiments=num_experiments, num_epochs=num_epochs)

    print(tab)

    # check number of files in repo == 2 for each experiment + 1 csv
    assert len([fname for fname in os.listdir(tmp_folder) if fname.endswith('.csv') or
                fname.endswith('.pkl') or fname.endswith('.h5')]) == num_experiments * 2 + 1

    shutil.rmtree(tmp_folder)



if __name__ == '__main__':
    pytest.main([__file__])