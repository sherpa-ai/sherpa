from __future__ import print_function
import tempfile
import shutil
import os
import pytest
import pickle as pkl
from sherpa.resultstable import ResultsTable
import os
import datetime
import sherpa
from sherpa.hyperparameters import DistributionHyperparameter as Hyperparameter
from sherpa.scheduler import LocalScheduler, SGEScheduler

@pytest.fixture
def test_dir():
    dirpath = tempfile.mkdtemp()
    yield dirpath
    print("deleting tempdir")
    shutil.rmtree(dirpath)


def test_resultstable(test_dir):
    results_table = ResultsTable(dir=test_dir)
    results_table._set(index=0, loss=0.1, hp={'Int': 1, 'Float': 0.1,
                                              'Str': 'abc'})
    with open(os.path.join(test_dir, 'results.csv'), 'r') as csv:
        next(csv)
        Epochs, Float, History, ID, Int, Loss, Pending, Str = next(csv).split(',')
        assert Epochs == '0'
        assert Float == '0.1'
        assert ID == '0'
        assert Int == '1'
        assert Str == 'abc\n'


def main(modelfile, historyfile, hp={}, epochs=1, verbose=2):
    """
    Test main function designed to break for one of the parameters.
    """
    result = 1/(2 - hp.get('p', 0))
    print("Result {}".format(result))
    # Update history and save to file.
    with open(historyfile, 'wb') as fid:
        pkl.dump({'loss': [result]}, fid)


def test_exit_after_failed_process(test_dir):
    """
    Test that main loop exits if a process fails
    """
    alg = sherpa.algorithms.Iterate(epochs=1,
                                    hp_iter=[{'p': 1}, {'p': 2},
                                             {'p': 3}])
    sched = LocalScheduler()
    rval = sherpa.optimize(filename=os.path.basename(__file__),
                           algorithm=alg, dir=test_dir,
                           overwrite=True, scheduler=sched, max_concurrent=2)