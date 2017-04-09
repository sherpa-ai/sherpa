from __future__ import print_function
from __future__ import absolute_import
import hobbit.experiment as experiment
from hobbit.utils.testing_utils import create_model, load_dataset
import tempfile
import shutil
import numpy as np
import pytest

@pytest.mark.run(order=4)
def test_experiment():
    """
    Runs tests on experiment class

    """
    tmp_folder = tempfile.mkdtemp(prefix='test_repo')

    # prereq's
    total_epochs = 5
    batch_size = 128
    hparams = {'lr': 0.01, 'num_units': 100}
    (x_train, y_train), (x_test, y_test) = load_dataset()

    # train experiment and compare
    epochs = 1
    exp = experiment.Experiment(path=tmp_folder, name='1_5', model=create_model(hparams))
    best_performance = exp.fit(x=x_train, y=y_train, epochs=epochs,
                               batch_size=batch_size, validation_data=(x_test, y_test))
    del exp

    for i in range(total_epochs-1):
        exp = experiment.Experiment(path=tmp_folder, name='1_5')
        best_performance, epochs_seen = exp.fit(x=x_train, y=y_train, epochs=epochs,
                                   batch_size=batch_size, validation_data=(x_test, y_test))
        del exp

    # train model in regular way for 5 epochs
    test_model = create_model(hparams)
    hist = test_model.fit(initial_epoch=0, x=x_train, y=y_train,
                        batch_size=batch_size, epochs=total_epochs,
                        verbose=1, validation_data=(x_test, y_test))

    assert np.isclose(min(hist.history['val_loss']), best_performance, rtol=0.02, atol=0.02)

    # experiment that doesn't exist
    with pytest.raises(IOError):
        experiment.Experiment(path=tmp_folder, name='1_6')

    # test that loading an experiment with deleted model file throws correct exception

    shutil.rmtree(tmp_folder)

if __name__ == '__main__':
    pytest.main([__file__])
