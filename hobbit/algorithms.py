from __future__ import absolute_import
from .schedulers import JobScheduler
from .resultstable import ResultsTable
from .hparam_generators import RandomGenerator
from . import Repository
import math
import os


class Hyperband(object):
    """
    A Hyperband instance initializes the entire pipeline needed to run a
    Hyperband hyperparameter optimization. The run() method is used to start
    the optimization.

    # Arguments
        model_function: a function that takes a dictionary of hyperparameters
            as its only argument and returns a compiled Keras model object with
            those hyperparameters
        hparam_ranges: a list of Hyperparameter objects
        repo_dir: the directory to store weights and results table in
        dataset: a dataset of the form ((x_train, y_train), (x_valid, y_valid))
            where x_, y_ are NumPy arrays
        generator_function: alternatively to dataset, a generator function can
            be passed or a tuple of generator functions. This is a function
            that returns a generator, not a generator itself. For a tuple
            the first item is the generator function for training, the second
            for validation.
        train_gen_args: arguments to be passed to generator_function when
            producing a training generator
        steps_per_epoch: number of batches for one epoch of training when
            using a generator
        valid_gen_args: arguments to be passed to generator_function when
            producing a validation generator
        validation_steps: number of batches for one epoch of validation when
            using a generator

    # Methods
    Runs the algorithm with **R** maximum epochs per stage and cut factor
    **eta** between stages.

    # run
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
    def __init__(self, model_function, hparam_ranges, loss='val_loss',
                 repo_dir='./hyperband_repository', dataset=None,
                 generator_function=None, train_gen_args=None, steps_per_epoch=None,
                 valid_gen_args=None, validation_steps=None):
        assert hparam_ranges
        assert model_function
        assert dataset or generator_function, "You need to pass either a dataset array or generator arguments"
        assert steps_per_epoch and validation_steps if generator_function else True,\
            "You need to pass the number of batches/steps per epoch for training and validation"

        os.makedirs(repo_dir) if not os.path.exists(repo_dir) else None

        self.results_table = ResultsTable(repo_dir)

        repo = Repository(model_function=model_function,
                          dataset=dataset,
                          results_table=self.results_table,
                          dir=repo_dir,
                          loss=loss,
                          generator_function=generator_function,
                          train_gen_args=train_gen_args,
                          steps_per_epoch=steps_per_epoch,
                          valid_gen_args=valid_gen_args,
                          validation_steps=validation_steps)

        self.scheduler = JobScheduler(repository=repo)
        # Note, if we pass a scheduler we still need to pass the repo to it

        self.hparam_gen = RandomGenerator(hparam_ranges)

    def run(self, R=20, eta=3):
        log_eta = lambda x: math.log(x) / math.log(eta)
        s_max = int(log_eta(R))
        B = (s_max + 1) * R

        for s in reversed(range(s_max + 1)):
            n = int(math.ceil(B / R / (s + 1) * eta ** s))  # initial number of configurations
            r = R * eta ** (-s)  # initial number of iterations to run configurations for

            for i in range(s + 1):
                # Run each of the n_i configs for r_i iterations and keep best n_i/eta
                n_i = int(n * eta ** (-i))
                r_i = int(round(r * eta ** (i)))

                if i == 0:
                    [self.scheduler.submit(run_id=(s, j), hparams=self.hparam_gen.next(), epochs=r_i) for j in range(n_i)]
                else:
                    [self.scheduler.submit(run_id=(s, T_j), epochs=r_i) for T_j in self.results_table.get_k_lowest_from_run(n_i, s)]

        return self.results_table._get_table()


class RandomSearch(Hyperband):
    """
    Base class for specific algorithms e.g. Hyper-band

    # Example:

    Hyperband(my_model, my_dataset, my_hparam_ranges, my_scheduler, repo_dir='./example/')

    Initializes entire pipeline

    init hparam_gen(hparam_ranges)
    init results(repo_dir)
    init repo(results, model_func, dataset, repo_dir)
    scheduler.set_repository(repo)
    """
    def __init__(self, **kwargs):
        super(RandomSearch, self).__init__(**kwargs)


    def run(self, num_experiments, num_epochs):
        """
        Args:
            num_experiments:
            num_epochs:

        Returns:

        """
        run = 1
        for id in range(num_experiments):
            self.scheduler.submit(run_id=(run, id), hparams=self.hparam_gen.next(), epochs=num_epochs)
            print(self.results_table._get_table())

        return self.results_table._get_table()
