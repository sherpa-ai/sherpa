from __future__ import print_function
from __future__ import absolute_import
from hobbit.core import Repository
from hobbit.resultstable import ResultsTable
from hobbit.utils.testing_utils import create_model, load_dataset, store_mnist_hdf5, get_hdf5_generator
import tempfile
import shutil
import numpy as np
import pytest
import os
import csv
import h5py


@pytest.mark.run(order=5)
def test_repository():
    tmp_folder = tempfile.mkdtemp(prefix='test_repo')

    results_table = ResultsTable(tmp_folder)

    (x_train, y_train), (x_test, y_test) = load_dataset()

    repo = Repository(model_function=create_model, dataset=(x_train, y_train), validation_data=(x_test, y_test),
                      results_table=results_table, dir=tmp_folder)

    hparams = {'lr': 0.01, 'num_units': 100}

    repo.train(run_id=(1, 1), hparams=hparams, epochs=2)
    repo.train(run_id=(1, 2), hparams=hparams, epochs=2)

    repo.train(run_id=(1, 1), epochs=3)
    repo.train(run_id=(1, 2), epochs=3)

    assert np.isclose(results_table.get((1, 1), parameter='Loss'), results_table.get((1, 2), parameter='Loss'), rtol=0.05, atol=0.05)

    # train model in regular way for 5 epochs
    total_epochs = 5
    batch_size = 128
    test_model = create_model(hparams)
    hist = test_model.fit(initial_epoch=0, x=x_train, y=y_train,
                          batch_size=batch_size, epochs=total_epochs,
                          verbose=1, validation_data=(x_test, y_test))

    assert np.isclose(results_table.get(run_id=(1, 1), parameter='Loss'), min(hist.history['val_loss']), rtol=0.05, atol=0.05)
    assert np.isclose(results_table.get(run_id=(1, 2), parameter='Loss'), min(hist.history['val_loss']), rtol=0.05, atol=0.05)

    # manually load tmp_folder/results.csv
    with open(os.path.join(tmp_folder, 'results.csv')) as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for i, line in enumerate(reader, start=1):
            print(line)
            run_id, epochs, hparams, id, val_loss, run = line
            assert int(epochs) == 5

    shutil.rmtree(tmp_folder)

@pytest.mark.run(order=6)
def test_repository_with_generator():
    tmp_folder = tempfile.mkdtemp(prefix='test_repo')

    hparams = {'lr': 0.01, 'num_units': 100}
    batch_size = 100

    results_table = ResultsTable(tmp_folder)

    path_to_hdf5 = store_mnist_hdf5(tmp_folder)

    with h5py.File(path_to_hdf5) as f:
        num_train_batches = np.ceil(f['x_train'].shape[0]/batch_size).astype('int')
        num_test_batches = np.ceil(f['x_test'].shape[0] / batch_size).astype('int')

        repo = Repository(model_function=create_model,
                          generator_function=get_hdf5_generator,
                          train_gen_args=(f['x_train'], f['y_train'], batch_size),
                          valid_gen_args={'x': f['x_test'], 'y': f['y_test'], 'batch_size': batch_size},
                          steps_per_epoch=num_train_batches,
                          validation_steps=num_test_batches,
                          results_table=results_table,
                          dir=tmp_folder)

        repo.train(run_id=(1, 1), hparams=hparams, epochs=2)
        repo.train(run_id=(1, 2), hparams=hparams, epochs=2)

        repo.train(run_id=(1, 1), epochs=3)
        repo.train(run_id=(1, 2), epochs=3)

        assert np.isclose(results_table.get((1, 1), parameter='Loss'), results_table.get((1, 2), parameter='Loss'), rtol=0.05, atol=0.05)

        # train model in regular way for 5 epochs
        total_epochs = 5
        batch_size = 128
        test_model = create_model(hparams)
        hist = test_model.fit_generator(generator=get_hdf5_generator(f['x_train'], f['y_train'],
                                                                     batch_size=batch_size),
                                        steps_per_epoch=num_train_batches,
                                        epochs=total_epochs,
                                        validation_data=get_hdf5_generator(f['x_test'], f['y_test'],
                                                                           batch_size=batch_size),
                                        validation_steps=num_test_batches)

        assert np.isclose(results_table.get(run_id=(1, 1), parameter='Loss'), min(hist.history['val_loss']), rtol=0.2, atol=0.2)
        assert np.isclose(results_table.get(run_id=(1, 2), parameter='Loss'), min(hist.history['val_loss']), rtol=0.2, atol=0.2)

    shutil.rmtree(tmp_folder)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_repository_with_generator()