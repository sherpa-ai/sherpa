import tempfile
import shutil
import os
from sherpa.resultstable import ResultsTable


def test_resultstable():
    dirpath = tempfile.mkdtemp()
    results_table = ResultsTable(dir=dirpath)
    results_table._set(index=0, loss=0.1, hp={'Int': 1, 'Float': 0.1,
                                              'Str': 'abc'})
    try:
        with open(os.path.join(dirpath, 'results.csv'), 'r') as csv:
            next(csv)
            line_idx, Epochs, Float, History, ID, Int, Loss, Pending, Str = next(csv).split(',')
            assert Epochs == '0'
            assert Float == '0.1'
            assert ID == '0'
            assert Int == '1'
            assert Str == 'abc\n'
    finally:
        shutil.rmtree(dirpath)


if __name__ == '__main__':
    test_resultstable()