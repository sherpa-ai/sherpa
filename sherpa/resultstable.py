import os
import glob
import pickle as pkl
import numpy as np
import pandas as pd
import abc
import copy

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
        self.loss = loss
        self.loss_summary = loss_summary or (lambda loss_list: loss_list[-1])
        return      
   
    def on_start(self, hp):
        '''
        Called before model training begins. 
        A new model instance with hyperparameters hp, and a new expid is 
        returned.
        '''
        # Create new row, return unique expid.
        expids = self.get_expids()
        if len(expids) == 0:
            expid = 0
        else:
            expid = max(expids) + 1
        self._set(expid=expid, hp=hp, pending=True)
        return expid

    def on_finish(self, expid, metricsfile):
        '''
        Update results table. Called by MainLoop.
        expid       = Unique expid for a model instantiation.
        metricsfile = Pickle file containing history dictionary.
        '''
        assert expid in self.get_expids(), 'Index {} not in {}'.format(expid, self.get_expids())
        try:
            with open(metricsfile, 'rb') as f:
                history = pkl.load(f)
        except OSError:
            raise ValueError("History file not found at {}. SHERPA requires"
                             "every experiment to store a"
                             "history file.".format(metricsfile))

        assert self.loss in history, 'Key {} not in {}'.format(self.loss, history.keys())
        epochs_seen   = len(history[self.loss])
        lowest_loss   = self.loss_summary(history[self.loss]) if epochs_seen>0 else np.inf
        self._set(expid=expid, loss=lowest_loss, epochs=epochs_seen, metricsfile=metricsfile, pending=False)
        return 

    @abc.abstractmethod
    def get_expids(self):
        ''' 
        Return list of unique experiment ids in the ResultsTable.
        Called by Algorithm.     
        '''
        raise NotImplementedError()
 
    @abc.abstractmethod
    def get_pending(self):
        '''
        Return list of unique expids in the ResultsTable that are pending.
        '''
        raise NotImplementedError()
   
    @abc.abstractmethod
    def get_best(self):
        ''' Return hp dictionary for the best result. '''
        raise NotImplementedError()

    @abc.abstractmethod
    def _set(self, expid, loss=None, epochs=None, hp=None, metricsfile=None, pending=False):
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
        self.csv_path = os.path.join(dir, 'results.csv') # Human-readable results.
        self.keys = ('ID', 'Loss', 'Epochs', 'History', 'Pending')
        self.dtypes = {'ID': np.int, 'Loss': np.float64, 'Epochs': np.int,
                       'History': np.str, 'Pending': np.bool}
        self.expid2hp = {}  # Stash hyperparameter dicts in original form.
        try:
            os.makedirs(dir)
        except:
            pass

        # TODO: Have _create_table load existing metricsfile data from specified dir.
        self._create_table()

        # Optional: load existing results into table.
        # Note that this is better than loading results.csv because this 
        # Ensures that the loss and loss_summary are calculated properly.
        if load_results is not None:
            if type(load_results) is str:
                if not os.path.isdir(load_results):
                    raise ValueError('load_results must be directory of pkl files or list. Could not find path {}'.format(load_results))
                files = glob.glob('{}/*.pkl'.format(load_results))
                print('Loading {} metric files into results table from {}/'.format(len(hfiles), load_results))
            else:
                files = load_results
            for f in files:
                self.load(metricsfile=f)
        return
    
    def load(self, metricsfile):
        ''' Load result from metricsfile.'''
        with open(metricsfile, 'rb') as f:
            history = pkl.load(f) 
        hp = history['hp']
        expid = self.on_start(hp=hp)
        self.on_finish(expid=expid, metricsfile=metricsfile)

    def _create_table(self):
        ''' Creates new, empty, table and saves it to disk.'''
        self.df = pd.DataFrame()
        self._save()
        
    def _load_csv(self):
        '''
        Loads table from disk and returns it.
        # Returns: pandas df
        '''
        try:
            self.df = pd.read_csv(self.csv_path, dtype=self.dtypes)
        except:
            print('Unable to read existing csv file at {} using'
                  'pandas'.format(self.csv_path))
            raise
 
    def _save(self):
        """
        Updates stored csv
        """
        if 'Loss' in self.df.columns:
            # Sort by loss for readability.
            self.df = self.df.sort_values(by=['Loss'])
        self.df.to_csv(self.csv_path, index=False)
       
    def _set(self, expid, loss=np.inf, epochs=0, hp=None, metricsfile=None, pending=False):
        """
        Sets a value for a model using expid as identifier and saves the hyperparameter description of it. 
        expid = Unique identifier for each model instantiation. Used to identify models that are paused/restarted.
        loss  = loss value.
        epochs = Total number of epochs that the model has been trained.
        metricsfile = File path.    
        """
        if len(self.df) == 0 or expid not in self.df['ID']:
            if hp is None:
                raise ValueError('Must provide hyperparameters for any new row.') 
            # Save raw hp dict.
            self.expid2hp[expid] = hp
            # Convert hyperparameter dict into resultstable friendly form. 
            hp = copy.deepcopy(hp)
            for key in hp:
                if type(hp[key]) in [list, dict]:
                    # Convert this hp value to string.
                    hp[key] = str(hp[key])
                if key not in self.dtypes:
                    self.dtypes[key] = type(hp[key])
            # New line.
            new_dict = {'ID': expid,
                        'Loss': loss,
                        'Epochs': epochs,
                        'History': metricsfile,
                        'Pending': pending}
            new_dict.update(hp)
            new_line = pd.DataFrame(new_dict, index=[expid])
            self.df = self.df.append(new_line)
        else:
            # Update previous result.
            ridx = self.df['ID'] == expid
            assert len(self.df.loc[ridx]) == 1, ridx
            self.df.loc[ridx, 'Loss']    = loss
            self.df.loc[ridx, 'Epochs']  = epochs
            self.df.loc[ridx, 'History'] = metricsfile
            self.df.loc[ridx, 'Pending'] = pending
            
        self._save()
    
    def get_expids(self):
        if len(self.df) > 0:
            return [i for i in self.df['ID']]
        else:
            return []
    
    def get_pending(self):
        if len(self.df) > 0:
            return [i for i in self.df['ID'] if self.df['Pending'].iloc[i] == True]
        else:
            return []


    def get_k_lowest(self, k=1, ignore_pending=True):
        """
        Gets the k models with lowest loss.

        # Args:
             k: Integer, number of id's to return
            
        # Returns:
             List of experiment ids.

        TODO: add more options, e.g. ignore repeats.
        """
        data = self.df.sort_values(by='Loss', ascending=True)
        if ignore_pending:
            data = data[data['Pending'] == False]
        if len(data) < k:
            raise ValueError('Tried to get top {} results but only found {} results.'.format(k, len(data)))
        assert len(data.index) >= k, len(data.index)
        expid_best = list(data.iloc[0:k]['ID'])
        return expid_best

    def get_best(self, ignore_pending=True, k=None):
        """
        Return hyperparameters of best experiment(s) so far.

        # Args:
            ignore_pending: Ignore any pending results.
            k: If None, return best hp dict. If k is int, return list.

        # Returns:
            Dict of hp, or list of dicts.
        """    
        if k is None:
            expid_best = self.get_k_lowest(k=1, ignore_pending=ignore_pending)
            return self.expid2hp[expid_best[0]]
        else:
            expid_best = self.get_k_lowest(k=k, ignore_pending=ignore_pending)
            return [self.expid2hp[expid_best[i]] for i in range(k)]

    def get_matches(self, hp):
        """
        Return expids of experiments that match hyperparameters,
        including those for which Pending=True. Note that we use 
        expid2hp instead of the dataframe because the complex 
        hyperparameters are transformed into strings for the df.
        # Args:
            hp: dict of hyperparameters. Possibly not complete.
        # Returns:
            List of expids that have these hyperparameters.
        """
        matches = []
        for expid, hpcombo in self.expid2hp.items():
            match = True
            for k,v in hp.items():
                if k not in hpcombo or hpcombo[k] != v:
                    match = False
                    break
            if match:
                matches.append(expid)
        return matches

