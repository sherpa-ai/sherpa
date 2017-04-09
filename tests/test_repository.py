from __future__ import print_function
from __future__ import absolute_import
from hobbit.core import Repository
from hobbit.resultstable import ResultsTable
from hobbit.utils.testing_utils import create_model, load_dataset
import tempfile
import shutil
import numpy as np
import pytest
import os
import csv

@pytest.mark.run(order=5)
def test_repository():
    tmp_folder = tempfile.mkdtemp(prefix='test_repo')

    results_table = ResultsTable(tmp_folder)

    (x_train, y_train), (x_test, y_test) = load_dataset()

    repo = Repository(model_function=create_model, dataset=((x_train, y_train), (x_test, y_test)),
                      results_table=results_table, dir=tmp_folder)

    hparams = {'lr': 0.01, 'num_units': 100}


    repo.train(run_id=(1, 1), hparams=hparams, epochs=2)
    repo.train(run_id=(1, 2), hparams=hparams, epochs=2)

    repo.train(run_id=(1, 1), epochs=3)
    repo.train(run_id=(1, 2), epochs=3)

    assert np.isclose(results_table.get_val_loss((1, 1)), results_table.get_val_loss((1, 2)), rtol=0.05, atol=0.05)


    # train model in regular way for 5 epochs
    total_epochs = 5
    batch_size = 128
    test_model = create_model(hparams)
    hist = test_model.fit(initial_epoch=0, x=x_train, y=y_train,
                        batch_size=batch_size, epochs=total_epochs,
                        verbose=1, validation_data=(x_test, y_test))

    assert np.isclose(results_table.get_val_loss(run_id=(1, 1)), min(hist.history['val_loss']), rtol=0.05, atol=0.05)
    assert np.isclose(results_table.get_val_loss(run_id=(1, 2)), min(hist.history['val_loss']), rtol=0.05, atol=0.05)

    # manually load tmp_folder/results.csv
    with open(os.path.join(tmp_folder, 'results.csv')) as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for i, line in enumerate(reader, start=1):
            print(line)
            run_id, epochs, hparams, id, run, val_loss = line
            assert int(epochs) == 5

    shutil.rmtree(tmp_folder)


if __name__ == '__main__':
    pytest.main([__file__])