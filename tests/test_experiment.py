from __future__ import print_function
from __future__ import absolute_import
import sherpa.experiment as experiment
from sherpa.utils.testing_utils import create_model, load_dataset, store_mnist_hdf5, get_hdf5_generator
import tempfile
import shutil
import numpy as np
import pytest
import h5py

# @pytest.mark.run(order=4)
# def test_experiment():
#     """
#     Runs tests on experiment class
#
#     """
#     tmp_folder = tempfile.mkdtemp(prefix='test_repo')
#
#     # prereq's
#     total_epochs = 5
#     batch_size = 128
#     hparams = {'lr': 0.01, 'num_units': 100}
#     (x_train, y_train), (x_test, y_test) = load_dataset()
#
#     # train experiment and compare
#     epochs = 1
#     exp = experiment.Experiment(path=tmp_folder, name='1_5', model=create_model(hparams))
#     best_performance = exp.fit(x=x_train, y=y_train, epochs=epochs,
#                                batch_size=batch_size, validation_data=(x_test, y_test))
#     del exp
#
#     for i in range(total_epochs-1):
#         exp = experiment.Experiment(path=tmp_folder, name='1_5')
#         best_performance, epochs_seen = exp.fit(x=x_train, y=y_train, epochs=epochs,
#                                    batch_size=batch_size, validation_data=(x_test, y_test))
#         del exp
#
#     # train model in regular way for 5 epochs
#     test_model = create_model(hparams)
#     hist = test_model.fit(initial_epoch=0, x=x_train, y=y_train,
#                         batch_size=batch_size, epochs=total_epochs,
#                         verbose=1, validation_data=(x_test, y_test))
#
#     assert np.isclose(min(hist.history['val_loss']), best_performance, rtol=0.02, atol=0.02)
#
#     # experiment that doesn't exist
#     with pytest.raises(IOError):
#         experiment.Experiment(path=tmp_folder, name='1_6')
#
#     # test that loading an experiment with deleted model file throws correct exception
#
#     shutil.rmtree(tmp_folder)

@pytest.mark.run(order=4)
def test_experiment_with_generator():
    """
    Runs tests on experiment class when the dataset is from a generator

    """
    tmp_folder = tempfile.mkdtemp(prefix='test_repo')

    # prereq's
    total_epochs = 5
    epochs = 1
    batch_size = 128
    hparams = {'lr': 0.01, 'num_units': 100}

    path_to_hdf5 = store_mnist_hdf5(tmp_folder)

    with h5py.File(path_to_hdf5) as f:
        num_train_batches = np.ceil(f['x_train'].shape[0]/batch_size).astype('int')
        num_test_batches = np.ceil(f['x_test'].shape[0] / batch_size).astype('int')

        # train experiment and compare
        exp = experiment.Experiment(path=tmp_folder, name='1_5', model=create_model(hparams))
        best_performance, epochs_seen = exp.fit(generator=get_hdf5_generator(f['x_train'], f['y_train'],
                                                                             batch_size=batch_size),
                                                steps_per_epoch=num_train_batches,
                                                epochs=epochs,
                                                validation_data=get_hdf5_generator(f['x_test'], f['y_test'],
                                                                                   batch_size=batch_size),
                                                validation_steps=num_test_batches)
        del exp

        for i in range(total_epochs-1):
            exp = experiment.Experiment(path=tmp_folder, name='1_5')
            best_performance, epochs_seen = exp.fit(generator=get_hdf5_generator(f['x_train'], f['y_train'],
                                                                                 batch_size=batch_size),
                                                    steps_per_epoch=num_train_batches,
                                                    epochs=epochs,
                                                    validation_data=get_hdf5_generator(f['x_test'], f['y_test'],
                                                                                       batch_size=batch_size),
                                                    validation_steps=num_test_batches)
            del exp

        # train model in regular way for 5 epochs
        test_model = create_model(hparams)
        hist = test_model.fit_generator(generator=get_hdf5_generator(f['x_train'], f['y_train'],
                                                                     batch_size=batch_size),
                                        steps_per_epoch=num_train_batches,
                                        epochs=total_epochs,
                                        validation_data=get_hdf5_generator(f['x_test'], f['y_test'],
                                                                           batch_size=batch_size),
                                        validation_steps=num_test_batches)

        assert np.isclose(min(hist.history['val_loss']), best_performance, rtol=0.02, atol=0.02)

    shutil.rmtree(tmp_folder)


if __name__ == '__main__':
    pytest.main([__file__])