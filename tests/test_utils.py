from __future__ import print_function
from __future__ import absolute_import
from hobbit.utils.testing_utils import store_mnist_hdf5, get_hdf5_generator
import tempfile
import shutil
import numpy as np
import pytest
import h5py


def test_generator_utils():
    tmp_folder = tempfile.mkdtemp(prefix='test_repo')

    print(tmp_folder)

    path_to_hdf5 = store_mnist_hdf5(tmp_folder)

    with h5py.File(path_to_hdf5) as f:
        test_batch_size = 100
        num_extra_batches = 10

        train_gen = get_hdf5_generator(f['x_train'], f['y_train'], batch_size=test_batch_size)

        x, y = next(train_gen)

        assert x.shape == (test_batch_size, 784)
        assert y.shape == (test_batch_size, 10)

        total_num_batches = np.ceil(f['x_train'].shape[0]/test_batch_size).astype('int')

        for i in range(total_num_batches+num_extra_batches):
            x_new, y_new = next(train_gen)
            assert not np.array_equal(x_new, x)
            x, y = x_new, y_new

    shutil.rmtree(tmp_folder)

    return

if __name__ == '__main__':
    pytest.main([__file__])