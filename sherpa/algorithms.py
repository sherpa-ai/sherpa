from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .resultstable import ResultsTable
from .hparam_generators import RandomGenerator
import math

class RandomSearch():
    """
    Simple random search.
    """
    def __init__(self, samples, epochs, hp_ranges, max_concurrent=10):
        self.samples        = samples
        self.epochs         = epochs
        self.hp_ranges      = hp_ranges
        self.hp_generator   = RandomGenerator(hp_ranges)
        self.max_concurrent = max_concurrent

        print('Sampling %d random hp combinations from %d dimensions.' % (
            samples, len(hp_ranges)))

    def next(self, results_table, pending):
        '''
        Examine current results and produce next experiment.
        Valid return values:
        1) run_id, hparams, epochs: Tells main loop to start this experiment.
        2) 'wait': Signal to main loop that we are waiting.
        3) 'stop': Signal to main loop that we are finished.
        '''
        assert isinstance(results_table, ResultsTable)
        assert isinstance(pending, dict)
        df     = results_table.get_table() # Pandas df
        assert isinstance(df.shape[0], int)
        assert isinstance(len(pending), int)
        run_id = '1_%d' % (len(pending)+df.shape[0]) # Results table requires run_ids in this form.
        if df.shape[0] == self.samples:
            return 'stop'
        elif len(pending) >= self.max_concurrent:
            return 'wait'
        elif len(pending)+df.shape[0] >= self.samples:
            return 'wait'
        else:
            return run_id, self.hp_generator.next(), self.epochs

class Hyperhack():
    '''
    Peter 2017
    '''
    def __init__(self, samples, epochs_per_stage, stages, hp_generator=RandomGenerator, survival=0.5, hp_ranges={}, max_concurrent=10, constraints=[]):
        self.samples     = samples # Initial number of samples.
        self.survival    = survival # Value in [0,1], population is reduced to this amount at each stage.
        self.epochs_per_stage = epochs_per_stage
        self.stages      = stages
        self.hp_ranges   = hp_ranges
        self.max_concurrent = max_concurrent
        #if hp_generator is None:
        #    hp_generator = RandomGenerator
        self.hp_generator = hp_generator(hp_ranges) # Initial sampling method for hyperparameters.
        
        # State of the algorithm.
        self.stage = 0
        self.population = []
        # Initialize population with hp combinations from generator that fit constraints.
        if len(constraints) == 0:
            self.population = [(('1_%d'%i), self.hp_generator.next()) for i in range(samples)] # Only one run.
        else:
            count = 0
            while count < samples:
                sample = self.hp_generator.next()
                sat = True
                for constraint in constraints:
                    sat = sat and constraint(sample)
                if sat:
                    self.population.append(('1_%d'%count, sample))
                    count += 1

    def next(self, results_table, pending):
        '''
        Examine current results and produce next experiment.
        Valid return values:
        1) run_id, hparams, epochs: Tells main loop to start this experiment.
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

        if len(pending) >= self.max_concurrent:
            return 'wait'

        if len(self.population) == 0:
            if len(pending) > 0:
                return 'wait' # Don't start next stage until everyone finishes previous stage.
            self.stage += 1
            if self.stage >= self.stages:
                return 'stop'
            else:
                k   = int(math.ceil(self.samples * self.survival**self.stage)) # Survivor number.
                run_ids = results_table.get_k_lowest_from_run(k, run=1) # Only 1 run.
                self.population = [(run_id, None) for run_id in run_ids] # Use empty hp to indicate restart training.
                print('\nStage %d/%d: %d survivors, %d epochs per stage.' % (self.stage, self.stages, k, self.epochs_per_stage))
                # Display best so far.
                run_id = results_table.get_k_lowest_from_run(k=1, run=1)[0]
                best   = {'ID':run_id}
                for k in ['Loss', 'Epochs', 'Hparams']:
                    best[k] = results_table.get(run_id=run_id, parameter=k)
                print('Best loss:%0.4f epochs:%d id:%s hp:%s' % (best['Loss'], best['Epochs'], run_id, best['Hparams']))

        run_id, hparams = self.population.pop(0)
        return run_id, hparams, self.epochs_per_stage

#
# class Hyperband():
#     '''
#     Hyperband
#     '''
#     def __init__(self, R, eta, hp_ranges, max_concurrent=10):
#         self.R = R
#         self.eta = eta
#         self.hp_ranges = hp_ranges
#         self.hp_generator = RandomGenerator(hp_ranges)
#         self.max_concurrent = max_concurrent
#
#         # Visualize schedule.
#         total_epochs = visualize_hyperband_params(R=self.R, eta=self.eta)
#
#         # State variables.
#         log_eta = lambda x: math.log(x) / math.log(eta)
#         s_max = int(log_eta(R))
#         B = (s_max + 1) * R
#         self.s = 0
#         self.i = 0
#         self.j = 1 # range()
#
#     def next(self, results_table, pending):
#         '''
#         Examine current results and produce next experiment.
#         Valid return values:
#         1) run_id, hparams, epochs: Tells main loop to start this experiment.
#         2) 'wait': Signal to main loop that we are waiting.
#         3) 'stop': Signal to main loop that we are finished.
#         '''
#
#         if len(pending) >= max_concurrent:
#             return 'wait'
#
#         log_eta = lambda x: math.log(x) / math.log(eta)
#         s_max = int(log_eta(R))
#         B = (s_max + 1) * R
#
#         for s in reversed(range(s_max + 1)):
#             n = int(math.ceil(B / R / (s + 1) * eta ** s))
#             r = R * eta ** (-s)
#
#             for i in range(s + 1):
#                 n_i = int(n * eta ** (-i))
#                 r_i = int(round(r * eta ** (i)))
#
#                 run = s_max - s + 1
#                 if i == 0:
#                     for j in range(1, n_i+1):
#                         if s==s_max and i==0 and j==1:
#                             self.estimate_time(self.scheduler.submit,
#                                                {'run_id': '{}_{}'.format(run,j),
#                                                 'hparams':
#                                                     self.hparam_gen.next(),
#                                                 'epochs': r_i},
#                                                total_epochs=total_epochs,
#                                                r_i=r_i)
#
#                         else:
#                             self.scheduler.submit(run_id='{}_{}'.format(run,
#                                                                         j),
#                                                   hparams=self.hparam_gen.next(),
#                                                   epochs=r_i)
#                 else:
#                     for run_id in self.results_table.get_k_lowest_from_run(n_i,
#                                                                         run):
#                         self.scheduler.submit(run_id=run_id, epochs=r_i)
#
#         return self.results_table.get_table()
#
#     @staticmethod
#     def estimate_time(f, args, total_epochs, r_i):
#         time, result = timedcall(f, args)
#
#         secs = total_epochs * time / r_i
#         hrs = secs // 3600
#         mins = (secs % 3600) // 60
#         print('-' * 100)
#         print('\nThe complete Hyperband optimization is '
#               'estimated to take {}hrs and {} '
#               'mins\n'.format(
#             hrs, mins))
#         print('-' * 100)
#
