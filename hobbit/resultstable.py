import pandas as pd
import numpy as np
import os


class ResultsTable(object):
    """
    Handles input/output of an underlying hard-disk stored .csv that stores the results
    """
    def __init__(self, dir):
        self.csv_path = os.path.join(dir, 'results.csv')
        self.keys = ('Run', 'ID', 'Hparams', 'Val Loss')
        df = self._create_table()
        self._save(df)
        return

    def get_k_lowest_from_run(self, k, run):
        """
        Args:
            k: number of id's to return
            run: refers to hyper-band run

        Returns:
            list of id's
        """
        df = self._get_table()
        df_run = df[df['Run'] == run]
        sorted_df_run = df_run.sort_values(by='Val Loss', ascending=True)
        return list(sorted_df_run['ID'][0:k])

    def set(self, run_id, val_loss, hparams=None):
        """
        Args:
            run_id:
            val_loss:
            hparams:

        Returns:

        """
        df = self._get_table()
        run, id = run_id
        if hparams:
            new_line = pd.DataFrame({key: [val] for key, val in zip(self.keys, (run, id, hparams, val_loss))},
                                    index=[self._get_idx(run, id)])
            df = df.append(new_line)
        else:
            df.set_value(index=self._get_idx(run, id), col='Val Loss', value=val_loss)
        self._save(df)

    def _get_idx(self, run, id):
        return '{}-{}'.format(run, id)

    def _create_table(self):
        """

        Returns: By default a pandas dataframe with columns:
        ID - RunID - Hparams - Val Loss

        """
        return pd.DataFrame(columns=self.keys)

    def _get_table(self):
        """

        Returns: pandas df

        """
        return self._load(self.csv_path)

    def _get(self, run_id):
        """
        Args:
            run_id:

        Returns:

        """
        run, id = run_id
        df = self._get_table()
        return df.ix[self._get_idx(run, id)]['Val Loss']

    def __getitem__(self, key):
        return self._get(key)

    def get_val_loss(self, run_id):
        return self._get(run_id)

    def _load(self, path):
        """
        Returns:
            pandas dataframe, loaded from on-disk csv
        """
        return pd.read_csv(path, index_col=0, dtype={'Run': np.int32,
                                                     'ID': np.int32,
                                                     'Val Loss': np.float64,
                                                     'Hparams': np.dtype('U')})

    def _save(self, df):
        """
        Updates stored csv
        Returns:

        """
        df.to_csv(self.csv_path)