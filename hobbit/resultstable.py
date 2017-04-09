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
        Gets the k models with lowest loss from a specific run. Note this is not necesarily the global minimum
        
        # Args:
            k: Integer, number of id's to return
            run: Integer, refers to hyper-band run

        # Returns:
            list of id's from this run
        """
        df = self._get_table()
        df_run = df[df['Run'] == run]
        sorted_df_run = df_run.sort_values(by='Val Loss', ascending=True)
        return list(sorted_df_run['ID'][0:k])

    def set(self, run_id, val_loss, hparams=None):
        """
        Sets a value for a model using (run, id) as identifier and saves the hyperparameter description of it. 
        
        # Args:
            run_id: Tuple, contains run and id numbers
            val_loss: float, validation loss value to set in table
            hparams:


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
        """
        Returns the run, id in a string with the format to be used in the table
        
        # Args 
            run: Integer, run number
            id: Integer, id number within the run

        # Returns:
           String with correct format for identification
        """
        return '{}-{}'.format(run, id)

    def _create_table(self):
        """
        Initializes a pandas dataframe with the set of keys of this object
        
        # Returns: 
            pandas dataframe

        """
        return pd.DataFrame(columns=self.keys)

    def _get_table(self):
        """
        Loads table from disk and returns it.
        # Returns: pandas df

        """
        return self._load(self.csv_path)

    def _get(self, run_id, parameter=None):
        """
        Returns parameter value of a model from the table
        Args:
            run_id: tuple, (run, id)
            parameter: string, name of the parameter to be returned

        Returns: validation loss 

        """
        run, id = run_id
        df = self._get_table()
        assert parameter is not None, "you must specify a parameter to get from the resultstable keys"
        assert parameter in self.keys(), \
            'parameter must match with one of the keys of resultstable, found {}'.format(parameter)
        return df.ix[self._get_idx(run, id)][parameter]

    def __getitem__(self, key):
        return self._get(key)

    def get_val_loss(self, run_id):
        return self._get(run_id, 'Val Loss')

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
        # Args:
            df: dataframe

        """
        df.to_csv(self.csv_path)
