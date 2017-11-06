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

    TODO: Implement option to iterate over specific hp combinations, 
          rather than all. But maybe easier to do this in terms of constraints?
    '''
    def __init__(self, hp_ranges=None, hp_iter=None):
        '''
        Inputs:
          hp_ranges = Dictionary mapping hp names to lists of values.
          hp_iter   = List of dictionaries mapping hp names to values, 
                      which allows us to iterate over combinations.
        '''
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
            self.nsamples = np.prod(np.array([len(h) for h in hp_ranges.values()]))  
        elif hp_iter is not None:
            assert isinstance(hp_iter, list)
            self.sampler = IterateSampler(hp_iter)
            self.nsamples = len(hp_iter) 
        else:
            raise ValueError('Iterate algorithm expects either hp_ranges or hp_iter.')
        print('Iterate Algorithm: Iterating over {} hp combinations.'.format(self.nsamples))

    def next(self, results_table):
        '''
        Examine current results and produce next experiment.
        Valid return values:
        1) 'wait': Signal to main loop that we are waiting.
        2) 'stop': Signal to main loop that we are finished.
        3) hp: Tells main loop to start this experiment.
        '''
        assert isinstance(results_table, ResultsTable)
        try:
            return self.sampler.next()
        except StopIteration:
            return 'stop'
        

class RandomSearch(AbstractAlgorithm):
    """
    Random Search over hyperparameter space.

    # Arguments
        samples (int): Number of trials to evaluate.
        hp_ranges (list): List of Hyperparameter objects.
    """
    def __init__(self, samples, hp_ranges):
        self.samples = samples
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
        1) hp: Tells main loop to start this experiment.
        2) 'wait': Signal to main loop that we are waiting.
        3) 'stop': Signal to main loop that we are finished.
        '''
        assert isinstance(results_table, ResultsTable)
        idxs = results_table.get_expids() # Pandas df
        if len(idxs) == self.samples:
            return 'stop'
        else:
            self.count += 1
            return self.sampler.next()

class LocalSearch(AbstractAlgorithm):
    '''
    Greedily try to improve best result by changing one hyperparameter at a time. 

    TODO: Generalize this to real-valued parameters.
    '''
    def __init__(self, hp_ranges, hp_init=None):
        '''
        Inputs:
          hp_ranges = Dictionary mapping hp names to lists of values.
          hp_init   = Run this first if not in results table. Otherwise randomly select. 
        '''
        assert all([type(v) == list for v in hp_ranges.values()]), 'All dict values should be lists of possible values: {}'.format(hp_ranges)
        self.hp_ranges = hp_ranges
        self.hp_init = hp_init # None if hp_init not specified.
        # Specify a RandomSampler for beginning. 
        temp = [Hyperparameter.fromlist(name, choices) for (name,choices) in hp_ranges.items()]
        self.randomsampler = RandomSampler(temp)

    def next(self, results_table):
        '''
        Examine current results and produce next experiment.
        Valid return values:
        1) 'wait': Signal to main loop that we are waiting.
        2) 'stop': Signal to main loop that we are finished.
        3) hp: Tells main loop to start this experiment.
        '''
        assert isinstance(results_table, ResultsTable)
        if self.hp_init is not None and len(results_table.get_matches(self.hp_init)) == 0:
            # Start this point first.
            print('Starting with {}'.format(self.hp_init))
            return self.hp_init
       
        # If ResultsTable is empty, suggest random. 
        if len(results_table.get_expids()) == 0:
            return self.randomsampler.next()
            
        # Try to find best result so far.
        try:
            hp_best = results_table.get_best(ignore_pending=True)
            # TODO: Handle ties. Right now it just grabs one at random.
        except ValueError:
            # No finished results yet. Select random point.
            return self.randomsampler.next()
 
        # Try to improve best result.
        for key in self.hp_ranges.keys():
            vals = self.hp_ranges[key]
            for val in vals:
                # Create new hyperparameter dict with all the same except val.
                hp_next = hp_best.copy()
                hp_next[key] = val
                if len(results_table.get_matches(hp=hp_next)) == 0:
                    # Haven't tried this hp combination yet.
                    return hp_next

        # Nothing to submit right now.
        if len(results_table.get_pending()) > 0:
            return 'wait'
        else:
            return 'stop'  

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
