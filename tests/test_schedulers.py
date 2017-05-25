"""
Lazy test suite, furnish out when more schedulers are available
"""
from __future__ import print_function
from __future__ import absolute_import
from sherpa.core import Repository
from sherpa.resultstable import ResultsTable
from sherpa.schedulers import JobScheduler
from sherpa.utils.testing_utils import create_model, load_dataset
import tempfile
import shutil
import numpy as np
import pytest
from sherpa.utils.testing_utils import read_nvidia_smi, gpu_exists

@pytest.mark.run(order=7)
def test_job_scheduler():
    tmp_folder = tempfile.mkdtemp(prefix='test_repo')

    results_table = ResultsTable(tmp_folder)

    (x_train, y_train), (x_test, y_test) = load_dataset()

    repo = Repository(model_function=create_model, dataset=(x_train, y_train), validation_data=(x_test, y_test),
                      results_table=results_table, dir=tmp_folder)

    scheduler = JobScheduler(repository=repo)

    hparams = {'lr': 0.01, 'num_units': 100}

    scheduler.submit(run_id=(1, 1), hparams=hparams, epochs=2)
    scheduler.submit(run_id=(1, 2), hparams=hparams, epochs=2)

    scheduler.submit(run_id=(1, 1), epochs=3)
    scheduler.submit(run_id=(1, 2), epochs=3)

    assert np.isclose(results_table.get((1, 1), parameter='Loss'), results_table.get((1, 2), parameter='Loss'), rtol=0.02, atol=0.02)

    shutil.rmtree(tmp_folder)


if __name__ == '__main__':
    pytest.main([__file__])