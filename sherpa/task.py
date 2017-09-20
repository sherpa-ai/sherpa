"""
Replacement for main function. Other class names to consider besides "Task": Experiment, Run, Job, Sample, etc.
"""
import abc
import os
from collections import defaultdict
import pickle as pkl
class Task(object):
    
    
    def __init__(self,train,valid,model_init):
        self.train = train
        self.valid = valid
        self.model_init=model_init
        
        
    def __call__(self,modelfile,historyfile,modelout=None,historyout=None,hp={},epochs=1,verbose=2):
    
        """
        ---------------------------------------------------------------------------
        EDIT THIS METHOD
        ---------------------------------------------------------------------------
        This main function is called by Sherpa. 
        Input:
            modelfile  = File containing model.
            historyfile= File containing dictionary of per-epoch results.
            hp         = Dictionary of hyperparameters.
            epochs     = Number of epochs to train this round.
            verbose    = Passed to keras.fit_generator.
        Output:
            No return value is given, but updates modelfile and historyfile.
        """
        if os.path.isfile(historyfile):
            # Resume training.
            assert os.path.isfile(modelfile)
            assert hp is None or len(hp) == 0
            model = self.load(modelfile)
            with open(historyfile, 'rb') as f:
                history = pkl.load(f)
            initial_epoch = len(history['loss']) # Assumes loss is list of length epochs.
        else:
            # Create new model.
            model   = self.model_init(hp=hp)
            history = defaultdict(list)
            initial_epoch = 0
    
        print('Running with {}'.format(str(hp)))
        
        # Update history and save to file.
        partialh = self.fit(model,initial_epoch,epochs,verbose)
        
        for k in partialh:
            history[k].extend(partialh[k])
            
        with open(historyout or historyfile, 'wb') as fid:
            pkl.dump(history, fid)
        # Save model file if we want to restart.
        self.save(model,modelout or modelfile)
    
        return
        
    @abc.abstractmethod
    def fit(self,initial_epoch,epochs):
        raise NotImplementedError
    
    @abc.abstractmethod
    def load(self,modelfile):
        raise NotImplementedError
        
    @abc.abstractmethod
    def save(self,model):
        raise NotImplementedError


class KerasTask(Task):
    
    def __init__(self,train,valid,model_init,steps_per_epoch = 100, validation_steps = 10):
        Task.__init__(self,train,valid,model_init)
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
    
    def load(self,modelfile):
        return keras.model.load(modelfile)
        
    def fit(self,model,initial_epoch,epochs,verbose):
        model.fit_generator(self.train, 
                    steps_per_epoch=self.steps_per_epoch,
                    validation_data = self.valid, 
                    validation_steps = self.validation_steps,
                    epochs = epochs + initial_epoch,
                    initial_epoch = initial_epoch,
                    verbose = verbose)
        return model.history.history
        
    def save(self,model,modelfile):
        model.save(modelfile)
        
