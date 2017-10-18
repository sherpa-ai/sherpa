import tempfile
import shutil
import os
import pytest
from sherpa.resultstable import ResultsTable

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
