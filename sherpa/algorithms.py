from __future__ import absolute_import
from __future__ import division
#from .schedulers import JobScheduler
from .resultstable import ResultsTable
from .hparam_generators import RandomGenerator
from .utils.monitoring_utils import visualize_hyperband_params, timedcall
#from . import Repository
import math
import os


class RandomSearch():
    """
    Simple random search.
    """
    def __init__(self, samples, epochs, hp_ranges, max_concurrent=10):
        self.samples        = samples
        self.epochs     = epochs
        self.hp_ranges      = hp_ranges 
        self.hp_generator   = RandomGenerator(hp_ranges)
        self.max_concurrent = max_concurrent
        
        print 'Sampling %d random hp combinations from %d dimensions.' % (samples, len(hp_ranges))

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

class Hyperband():
    """
    An Algorithm instance initializes the entire pipeline needed to run a
    hyperparameter optimization. The run() method is used to start
    the optimization.

    # Arguments
        model_function: a function that takes a dictionary of hyperparameters
            as its only argument and returns a compiled Keras model object with
            those hyperparameters
        hparam_ranges: a list of Hyperparameter objects
        repo_dir: the directory to store weights and results table in
        loss: which loss to optimize e.g. 'val_loss', 'val_mse' etc.
        dataset: a dataset of the form ((x_train, y_train), (x_valid, y_valid))
            where x_, y_ are NumPy arrays
        generator_function: alternatively to dataset, a generator function can
            be passed. This is a function that returns a generator, not a generator 
            itself.
        train_gen_args: arguments to be passed to generator_function when
            producing a training generator
        steps_per_epoch: number of batches for one epoch of training when
            using a generator
        validation_data: generator function for the validation data, not the generator
        valid_gen_args: arguments to be passed to generator_function when
            producing a validation generator
        validation_steps: number of batches for one epoch of validation when
            using a generator

    # Methods
    Runs the algorithm with **R** maximum epochs per stage and cut factor
    **eta** between stages.

    # run
    Depends on each optimization algorithm. For Hyperband this is:
        R: The maximum epochs per stage. Hyperband has multiple runs each of
            which goes through multiple stages to discard configurations. At each
            of those stages Hyperband will train for a total of R epochs
        eta: The cut-factor. After each stage Hyperband will reduce the number
            of configurations by this factor. The training
            iterations for configurations that move to the next stage increase
            by this factor

    # Example
    ```python
    def my_model(hparams):
        '''Keras model defintion returns compiled Keras model based on hparams'''
        return keras_model

    my_dataset = load_my_dataset()

    my_hparam_ranges = [Hyperparameter(name='learning_rate', distribution='log-uniform', distr_args=(0.0001, 0.1)),
                    Hyperparameter(name='activation', distribution='choice', distr_args=[('sigmoid', 'tanh', 'relu')]),
                    Hyperparameter(name='dropout', distribution='uniform', distr_args=(0., 1.))]


    hband = Hyperband(model_function=my_model,
                    dataset=my_dataset,
                    hparam_ranges=my_hparam_ranges,
                    repo_dir='./my_test_repo')

    results = hband.run(R=20, eta=3)
    ```
    """
    def __init__(self, R, eta, hp_ranges, max_concurrent=10):
        self.R = R
        self.eta = eta
        self.hp_ranges = hp_ranges 
        self.hp_generator = RandomGenerator(hp_ranges)
        self.max_concurrent = max_concurrent
        
        total_epochs = visualize_hyperband_params(R=self.R, eta=self.eta)

    def next(self, results_table, pending):
        '''
        Examine current results and produce next experiment.
        Valid return values:
        1) run_id, hparams, epochs: Tells main loop to start this experiment. 
        2) 'wait': Signal to main loop that we are waiting.
        3) 'stop': Signal to main loop that we are finished.
        '''
        if len(pending) >= max_concurrent:
            return 'wait'
        
        
        

        log_eta = lambda x: math.log(x) / math.log(eta)
        s_max = int(log_eta(R))
        B = (s_max + 1) * R

        for s in reversed(range(s_max + 1)):
            n = int(math.ceil(B / R / (s + 1) * eta ** s))
            r = R * eta ** (-s)

            for i in range(s + 1):
                n_i = int(n * eta ** (-i))
                r_i = int(round(r * eta ** (i)))

                run = s_max - s + 1
                if i == 0:
                    for j in range(1, n_i+1):
                        if s==s_max and i==0 and j==1:
                            self.estimate_time(self.scheduler.submit,
                                               {'run_id': '{}_{}'.format(run,j),
                                                'hparams':
                                                    self.hparam_gen.next(),
                                                'epochs': r_i},
                                               total_epochs=total_epochs,
                                               r_i=r_i)

                        else:
                            self.scheduler.submit(run_id='{}_{}'.format(run,
                                                                        j),
                                                  hparams=self.hparam_gen.next(),
                                                  epochs=r_i)
                else:
                    for run_id in self.results_table.get_k_lowest_from_run(n_i,
                                                                        run):
                        self.scheduler.submit(run_id=run_id, epochs=r_i)

        return self.results_table.get_table()

    @staticmethod
    def estimate_time(f, args, total_epochs, r_i):
        time, result = timedcall(f, args)

        secs = total_epochs * time / r_i
        hrs = secs // 3600
        mins = (secs % 3600) // 60
        print('-' * 100)
        print('\nThe complete Hyperband optimization is '
              'estimated to take {}hrs and {} '
              'mins\n'.format(
            hrs, mins))
        print('-' * 100)

