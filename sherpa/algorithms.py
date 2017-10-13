from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .hyperparameters import DistributionHyperparameter as Hyperparameter
from .resultstable import ResultsTable
from .samplers import RandomSampler, IterateSampler, GridSearch
import math
import numpy as np
import abc

class AbstractAlgorithm(object):
    @abc.abstractmethod
    def next(self, results_table):
        ''' 
        Abstract method implemented by Algorithms. Algorithm can assume
        that a hp combo that is emitted will be run by the mainloop --- 
        I.E. the mainloop/scheduler are responsible for scheduling 
        and error handling. 
        
        Input:
            results_table = Object of type ResultsTable.
        Output:
            'wait' = Tells MainLoop to wait for some time, so that pending 
                     results can finish.
            OR
            'stop' = Tells MainLoop that algorithm is done.
            OR
            hp, epochs = Tuple, with 'hp' a dict of hyperparameters to train, 
                         and for 'epochs' epochs.
            OR 
            index, epochs = Tuple, with index a int key for identifying model 
                            instance in results_table to continue training.

        Ideas:
        - Should we let Algorithm change hyperparameters of existing models?
        '''
        raise NotImplementedError()

class Iterate(AbstractAlgorithm):
    '''
    Simply iterate over all combinations in discrete hp space, then stop.
    '''
    def __init__(self, epochs=1, hp_ranges=None, hp_iter=None):
        '''
        Inputs:
          hp_ranges = Dictionary mapping hp names to lists of values.
          hp_iter   = List of dictionaries mapping hp names to values, 
                      which allows us to iterate over combinations.
        '''
        self.epochs  = epochs
        if hp_ranges is not None:
            if hp_iter is not None:
                raise NotImplementedError('TODO: combine hp_ranges with hp_iter in Iterate algorithm.')
            if isinstance(hp_ranges, dict):
                assert all([type(v) == list for v in hp_ranges.values()]), 'All dict values should be lists: {}'.format(hp_ranges)
                self.hp_ranges = [Hyperparameter.fromlist(name, choices) for (name,choices) in hp_ranges.items()]
            else:
                self.hp_ranges = hp_ranges
            # TODO: do we need to keep hp_ranges as a instance object?
            self.sampler = GridSearch(self.hp_ranges) # Iterate over all combinations of hp.
        elif hp_iter is not None:
            assert isinstance(hp_iter, list)
            #temp = [Hyperparameter.fromlist(name, [choice]) for (name, choice) in hp_iter.items()]
            #self.sampler = GridSearch(temp)
            self.sampler = IterateSampler(hp_iter)    
        else:
            raise ValueError('Iterate algorithm expects either hp_ranges or hp_iter.')

    def next(self, results_table):
        '''
        Examine current results and produce next experiment.
        Valid return values:
        1) 'wait': Signal to main loop that we are waiting.
        2) 'stop': Signal to main loop that we are finished.
        3) hp, epochs: Tells main loop to start this experiment.
        4) index, epochs: Tells main loop to resume this experiment.
        '''
        assert isinstance(results_table, ResultsTable)
        try:
            return self.sampler.next(), self.epochs
        except StopIteration:
            return 'stop'
        

class RandomSearch(AbstractAlgorithm):
    """
    Random Search over hyperparameter space.
    """
    def __init__(self, samples, epochs, hp_ranges):
        self.samples = samples
        self.epochs  = epochs
        self.count   = 0
        if isinstance(hp_ranges, dict):
            self.hp_ranges = [Hyperparameter.fromlist(name, choices) in name, choices in hp_ranges.items()]
        else:
            self.hp_ranges   = hp_ranges
        self.sampler   = RandomSampler(hp_ranges)

        print('Sampling %d random hp combinations from %d dimensions.' % (
            samples, len(hp_ranges)))

    def next(self, results_table):
        '''
        Examine current results and produce next experiment.
        Valid return values:
        1) hp, epochs: Tells main loop to start this experiment.
        1) index, epochs: Tells main loop to start this experiment.
        2) 'wait': Signal to main loop that we are waiting.
        3) 'stop': Signal to main loop that we are finished.
        '''
        assert isinstance(results_table, ResultsTable)
        idxs = results_table.get_indices() # Pandas df
        if len(idxs) == self.samples:
            return 'stop'
        else:
            self.count += 1
            return self.sampler.next(), self.epochs

class Hyperhack(AbstractAlgorithm):
    '''
    Successive halving variant. 
    Peter 2017
    '''
    def __init__(self, samples, epochs_per_stage, stages, sampler=RandomSampler, survival=0.5, hp_ranges={},  constraints=[]):
        self.samples     = samples # Initial number of samples.
        self.survival    = survival # Value in [0,1], population is reduced to this amount at each stage.
        self.epochs_per_stage = epochs_per_stage
        self.stages      = stages
        self.hp_ranges   = hp_ranges
        #if sampler is None:
        #    sampler = RandomSampler
        self.sampler = sampler(hp_ranges) # Initial sampling method for hyperparameters.
        
        # State of the algorithm.
        self.stage = 0
        self.population = []
        # Initialize population with hp combinations from generator that fit constraints.
        if len(constraints) == 0:
            self.population = [(('1_%d'%i), self.sampler.next()) for i in range(samples)] # Only one run.
        else:
            count = 0
            while count < samples:
                sample = self.sampler.next()
                sat = True
                for constraint in constraints:
                    sat = sat and constraint(sample)
                if sat:
                    self.population.append((count, sample))
                    count += 1

    def next(self, results_table):
        '''
        Examine current results and produce next experiment.
        Valid return values:
        1) index, hp, epochs: Tells main loop to start this experiment.
        2) 'wait': Signal to main loop that we are waiting.
        3) 'stop': Signal to main loop that we are finished.

        TODO: 
        1) The algorithm might want access to more information, e.g. the model and the full history.
           Therefore, I think the results table should include the modelfile and historyfile.

        '''
        if self.stage == 0 and len(self.population) == self.samples:
            print('\nStage %d/%d: %d samples, %d epochs per stage.' % (
                self.stage, self.stages, len(self.population),
                self.epochs_per_stage))

        if len(self.population) == 0:
            pending = results_table.get_pending()
            if len(pending) > 0:
                return 'wait' # Don't start next stage until everyone finishes previous stage.
            self.stage += 1
            if self.stage >= self.stages:
                return 'stop'
            else:
                k   = int(math.ceil(self.samples * self.survival**self.stage)) # Survivor number.
                indices = results_table.get_k_lowest(k) # Only 1 run.
                self.population = [(index, None) for index in indices] # Use empty hp to indicate restart training.
                print('\nStage %d/%d: %d survivors, %d epochs per stage.' % (self.stage, self.stages, k, self.epochs_per_stage))
                # Display best so far.
                best = results_table.get_best()
                print('Best loss:%0.4f epochs:%d index:%s hp:%s' % (best['Loss'], best['Epochs'], best['ID'], best['HP']))

        index, hp = self.population.pop(0)
        return index, hp, self.epochs_per_stage