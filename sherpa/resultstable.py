import os
import glob
import pickle as pkl
import numpy as np
import pandas as pd
import abc

class AbstractResultsTable(object):
    ''' 
    Required methods for MainLoop and Algorithms. Additional functionality 
    may be required for certain Algorithms. Implementation details may vary.
    '''
    def __init__(self, loss='loss', loss_summary=None):
        '''
        loss = Key in history to be minimized, E.G. 'loss', 'kl', or 'mse'.
        loss_summary = Function for summarizing loss from list, E.G. np.min.
        '''
        self.loss         = loss
        self.loss_summary = loss_summary or (lambda loss_list: loss_list[-1])
        return      
   
    def on_start(self, index=None, hp=None):
        '''
        Called before model training begins. If index is None, then this is
        a new model instance with hyperparameters hp, and a new index is 
        returned. Otherwise, index should be the index of a model to keep 
        training.
        '''
        if index is not None:
            # Resume training.
            assert hp is None
            assert index in self.get_indices()
            self._set(index=index, pending=True)
            return
        else:
            # Create new row, return unique index.
            indices = self.get_indices()
            if len(indices) == 0:
                index = 0
            else:
                index = max(indices) + 1
            self._set(index=index, hp=hp, pending=True)
            return index

    def on_finish(self, index, historyfile):
        '''
        Update results table. Called by MainLoop.
        index       = Unique index for a model instantiation.
        historyfile = Pickle file containing history dictionary.
        '''
        assert index in self.get_indices(), 'Index {} not in {}'.format(index, self.get_indices)
        with open(historyfile, 'rb') as f:
            history = pkl.load(f)
        assert self.loss in history, 'Key {} not in {}'.format(self.loss, history.keys())
        lowest_loss   = self.loss_summary(history[self.loss])
        epochs_seen   = len(history[self.loss])
        self._set(index=index, loss=lowest_loss, epochs=epochs_seen, historyfile=historyfile, pending=False)
        return 

    @abc.abstractmethod
    def get_indices(self):
        ''' 
        Return list of unique indices in the ResultsTable.
        Called by Algorithm.     
        '''
        raise NotImplementedError()
 
    @abc.abstractmethod
    def get_pending(self):
        '''
        Return list of unique indices in the ResultsTable that are pending.
        '''
        raise NotImplementedError()
   
    @abc.abstractmethod
    def get_best(self):
        ''' Return dictionary of info about the best result. '''
        raise NotImplementedError()

    @abc.abstractmethod
    def _set(self, index, loss=None, epochs=None, hp=None, historyfile=None, pending=False):
        ''' Set values in results table. '''
        raise NotImplementedError()
    
class ResultsTable(AbstractResultsTable):
    """
    Simple implementation of AbstractResultsTable.
    Uses pandas data frame to store results, and updates results.csv after every change.
    
    ID     = Unique ID for each model instantiation.
    Loss   = Current loss value.
    Epochs = Number of training epochs.
    Repeat = Is this model a repeat of another model. 
    HP     = Dictionary of hyperparameters.
    """
    def __init__(self, dir='./', loss='loss', loss_summary=None, load_results=None):
        '''
        dir  = Path where files can be saved.
        loss = Key in history to be minimized, E.G. 'loss', 'kl', or 'mse'.
        loss_summary = Function for summarizing loss from list, E.G. np.min.
        '''
        super(ResultsTable, self).__init__(loss=loss, loss_summary=loss_summary)
 
        self.dir = dir
        self.csv_path  = os.path.join(dir, 'results.csv') # Human-readable results.
        self.keys = ('ID', 'Loss', 'Epochs', 'HP', 'History', 'Pending')
        try:
            os.makedirs(dir)
        except:
            pass
        
        #if os.path.isfile(self.csv_path):
        #    print('WARNING: Overwriting results file at {}'.format(self.csv_path))
        # Create new table even if one already exists, as loss_summary function might have changed.
        # TODO: Have _create_table load existing historyfile data from specified dir.
        self._create_table()

        # Optional: load existing results into table.
        # Note that this is better than loading results.csv because this 
        # ensures that the loss and loss_summary are calculated properly.
        if load_results is not None:
            assert os.path.isdir(load_results), 'Could not find path {}'.format(load_results)
            hfiles = glob.glob('{}/*_history.pkl'.format(load_results))
            print('Loading {} history files into results table from {}/'.format(len(hfiles), load_results))
            for f in hfiles:
                #index = int(os.path.basename(f).split('_')[0])
                self.load(historyfile=f)
        return
    
    def load(self, historyfile):
        ''' Load result from historyfile.'''
        with open(historyfile, 'rb') as f:
            history = pkl.load(f) 
        hp = history['hp']
        index = self.on_start(hp=hp)
        self.on_finish(index=index, historyfile=historyfile)

    def _create_table(self):
        ''' Creates new, empty, table and saves it to disk.'''
        self.df = pd.DataFrame(columns=self.keys, )
        self._save()
        
    def _load_csv(self):
        '''
        Loads table from disk and returns it.
        # Returns: pandas df
        '''
        try:
            self.df = pd.read_csv(self.csv_path)
        except:
            print('Unable to read existing csv file at {} using pandas'.format(self.csv_path))
            raise
 
    def _save(self):
        """
        Updates stored csv
        """
        self.df.to_csv(self.csv_path)
       
    def _set(self, index, loss=np.inf, epochs=0, hp=None, historyfile=None, pending=False):
        """
        Sets a value for a model using index as identifier and saves the hyperparameter description of it. 
        index = Unique identifier for each model instantiation. Used to identify models that are paused/restarted.
        loss  = loss value.
        epochs = Total number of epochs that the model has been trained.
        historyfile = File path.    
        """ 
        if index in self.df.index:
            # Update previous result.
            self.df.set_value(index=index, col='Loss', value=loss)
            self.df.set_value(index=index, col='Epochs', value=epochs)
            self.df.set_value(index=index, col='History', value=historyfile)
            self.df.set_value(index=index, col='Pending', value=pending)
        else:
            # New line.
            new_line = pd.DataFrame([[index, loss, epochs, hp, historyfile, pending]], index=[index], columns=self.keys)
            self.df = self.df.append(new_line)
        self._save()
    
    def get_indices(self):
        return [i for i in self.df.index]

    def get_k_lowest(self, k):
        """
        Gets the k models with lowest global loss.

        # Args:
             k: Integer, number of id's to return

        # Returns:
             list with model ids

        TODO: add more options, e.g. ignore repeats.
        """
        assert len(self.df.index) >= k, len(self.df.index)
        df_sorted = self.df.sort_values(by='Loss', ascending=True)
        data = df_sorted.iloc[0:k]
        return data 

    def get_best(self):
        ''' Return values for best model so far.'''    
        data = self.get_k_lowest(k=1)
        bestdict = dict(zip(self.keys, [data[k].iloc[0] for k in self.keys]))
        return bestdict
   
    def update_hist2loss(self, hist2loss):
        ''' 
        Change hist2loss function, then update csv from historyfiles.
        '''
        self.hist2loss = hist2loss
        for index in self.df.index:
            historyfile = self.df.iloc[index]['History']
            self.update(index, historyfile)
        
                    

class ResultsTableOld(AbstractResultsTable):
    """
    DEPRECATED
    
    Handles input/output of an underlying hard-disk stored .csv that stores the results
    """
    def __init__(self, dir='./', loss='loss', overwrite=False):
        super(ResultsTable, self).__init__(dir=dir, loss=loss)
        self.keys = ('Run', 'ID', 'Hparams', 'Loss', 'Epochs')
        if overwrite:
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
        df = self.get_table()
        df_run = df[df['Run'] == run]
        sorted_df_run = df_run.sort_values(by='Loss', ascending=True)
        return list(sorted_df_run.index[0:k])

    def sample_k_ids_from_run(self, k, run, temperature=1.):
        """
        Samples k models with  probability proportional to inverse loss
        from a specific run. Note this is not necesarily the global minimum

        # Args:
            k: Integer, number of id's to return
            run: Integer, refers to hyper-band run

        # Returns:
            list of id's from this run
        """
        df = self.get_table()
        df_run = df[df['Run'] == run]
        # p = 1/df_run['Loss']
        p = np.exp(-df_run['Loss']/temperature)
        sampled_ids = np.random.choice(df_run.index, size=k, replace=False,
                                       p=p/np.sum(p))
        return list(sampled_ids)


    def get_k_lowest(self, k):
        """
        Gets the k models with lowest global loss.

        # Args:
             k: Integer, number of id's to return

        # Returns:
             list with run-id strings to identify models 
        """
        df = self.get_table()
        sorted_df_run = df.sort_values(by='Loss', ascending=True)
        ids = list(sorted_df_run['ID'][0:k])
        runs = list(sorted_df_run['Run'][0:k])
        id_run = []
        for row_id in zip(ids, runs):
            id_run.append(self._get_idx(row_id[0], row_id[1]))
        return id_run

    def set(self, run_id, loss, epochs, hp=None):
        """
        Sets a value for a model using (run, id) as identifier and saves the hyperparameter description of it. 
        
        # Args:
            run_id: Tuple, contains run and id numbers
            loss: float, e.g. validation loss value to set in table
            hp:


        """
        df = self.get_table()
        run, id = [int(num) for num in run_id.split('_')]
        if hp:
            new_line = pd.DataFrame({key: [val] for key, val in zip(self.keys, (run, id, hp, loss, epochs))},
                                    index=[run_id])
            df = df.append(new_line)
        else:
            df.set_value(index=run_id, col='Loss', value=loss)
            df.set_value(index=run_id, col='Epochs', value=epochs)
        self._save(df)

    def set_value(self, run_id, col, value):
        df = self.get_table()
        df.set_value(index=run_id, col=col, value=value)
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
        return '{}_{}'.format(run, id)

    def _create_table(self):
        """
        Initializes a pandas dataframe with the set of keys of this object
        
        # Returns: 
            pandas dataframe

        """
        return pd.DataFrame(columns=self.keys)

    def get_table(self):
        """
        Loads table from disk and returns it.
        # Returns: pandas df

        """
        return pd.read_csv(self.csv_path, index_col=0, dtype={'Run': np.int32,
                                                     'Epochs': np.int32,
                                                     'ID': np.int32,
                                                     'Loss': np.float64,
                                                     'Hparams': np.dtype('U')})

    def get(self, run_id, parameter=None):
        """
        Returns parameter value of a model from the table
        Args:
            run_id: tuple, (run, id)
            parameter: string, name of the parameter to be returned

        Returns: validation loss 

        """
        df = self.get_table()
        assert parameter is not None, "you must specify a parameter to get" \
                                      "from the resultstable keys"
        assert parameter in self.keys, \
            'parameter must match with one of the keys of resultstable,' \
            'found {}'.format(parameter)
        return df.ix[run_id][parameter]

    def _save(self, df):
        """
        Updates stored csv
        # Args:
            df: dataframe

        """
        df.to_csv(self.csv_path)

    def get_hp_df(self, as_design_matrix=False):
        df = self.get_table()
        hparam_df = pd.DataFrame([eval(item) for item in df['Hparams']])
        return hparam_df if not as_design_matrix or hparam_df.empty else \
            pd.get_dummies(hparam_df, drop_first=True)

    def get_column(self, key='Loss'):
        df = self.get_table()
        return df[key]
