from __future__ import absolute_import
from .experiment import Experiment
from .resultstable import ResultsTable


class Hyperparameter(object):
    """
    A Hyperparameter instance captures the information about each of the hyperparameters to be optimized.
    
    # Arguments
        name: name of the hyperparameter. This will be used in the Model creation
        distr_args: List, Tuple or Dictionary, it used by the distribution function as arguments. 
                    In the default case of a Uniform distribution these refer to the minimum and maximum values from 
                    which to sample from. In general these are the  arguments taken as input by the corresponding numpy 
                    distribution function.
        distribution: String, name of the distribution to be used for sampling
                    the values. Must be numpy.random compatible. Exception is
                    'log-uniform' which samples uniformly between low and high
                    on a log-scale. Uniform distribution is used as
                    default.
    
    # Examples
        ```python
        Hyperparameter('learning_rate', distr_args=(0.0001, 0.1), distribution='log-uniform'),
        Hyperparameter('learning_rate', distr_args={low: 0.0001, high: 0.1}, distribution='uniform'),
        Hyperparameter('activation', distr_args=[('sigmoid', 'tanh', 'relu')], distribution='choice')
        ```
    
    """
    def __init__(self, name, distr_args, distribution='uniform'):
        assert isinstance(name, str), 'name should be a string, found {}'.format(str(name.__class__))
        assert (isinstance(distr_args, list) or isinstance(distr_args, tuple) or isinstance(distr_args, dict)), \
            'distr_args should be a dictionary, tuple or list, found {}'.format(str(distr_args.__class__))
        assert isinstance(distribution, str), 'distribution should be a string, found {}'.format(
            str(distribution.__class__))

        self.name = name
        self.distr_args = distr_args
        self.distribution = distribution
        return


def run_id_to_str(run_id):
    return '{}-{}'.format(run_id[0], run_id[1])


class Repository(object):
    """
    A repository instance handles Experiment objects

    Submits results into the ResultsTable

    Receives training queries from the Scheduler
    """
    def __init__(self, model_function, results_table, dir='./', dataset=None,
                 generator_function=None, train_gen_args=None, steps_per_epoch=None,
                 valid_gen_args=None, validation_steps=None):
        assert isinstance(results_table, ResultsTable)
        assert isinstance(dataset, tuple) if dataset else True
        assert isinstance(dir, str)

        self.repo_dir = dir
        self.model_function = model_function
        self.dataset = dataset
        self.results_table = results_table
        self.generator_function = generator_function
        self.train_gen_args = train_gen_args
        self.steps_per_epoch = steps_per_epoch
        self.valid_gen_args = valid_gen_args
        self.validation_steps = validation_steps
        return

    def train(self, run_id, hparams=None, epochs=1):
        """
        Trains for one iteration, updates results and saves
        Args:
            run_id: (run, id)
            hparams: dict
            epochs: integer

        Returns:

        """
        print("Training model {} in run {}:".format(run_id[1], run_id[0]))
        if hparams:
            print(''.join(['{}: {:.4}\t\t'.format(key, str(hparams[key])) for key in hparams]))

        exp = self._get_experiment(run_id=run_id, hparams=hparams)

        if self.dataset:
            lowest_val_loss, epochs_seen = exp.fit(x=self.dataset[0][0], y=self.dataset[0][1], epochs=epochs,
                                                   batch_size=100, validation_data=self.dataset[1])
        else:
            if isinstance(self.generator_function, tuple):
                assert len(self.generator_function) == 2, "Generator function tuple needs to be length 2, one " \
                                                          "training and one testing."
                train_gen_function = self.generator_function[0]
                valid_gen_function = self.generator_function[1]
            else:
                train_gen_function = self.generator_function
                valid_gen_function = self.generator_function

            if isinstance(self.train_gen_args, dict):
                gen = train_gen_function(**self.train_gen_args)
            elif isinstance(self.train_gen_args, tuple) or isinstance(self.train_gen_args, list):
                gen = train_gen_function(*self.train_gen_args)
            elif self.train_gen_args is None:
                gen = train_gen_function()
            else:
                raise TypeError("Please provide generator arguments as dictionary, list or tuple."
                                "Found: " + str(self.train_gen_args) + " of type " + type(self.train_gen_args))

            if isinstance(self.valid_gen_args, dict):
                valid_gen = valid_gen_function(**self.valid_gen_args)
            elif isinstance(self.valid_gen_args, tuple) or isinstance(self.valid_gen_args, list):
                valid_gen = valid_gen_function(*self.valid_gen_args)
            elif self.valid_gen_args is None:
                valid_gen = valid_gen_function()
            else:
                raise TypeError("Please provide generator arguments as dictionary, list or tuple."
                                "Found: " + str(self.valid_gen_args) + " of type " + type(self.valid_gen_args))

            lowest_val_loss, epochs_seen = exp.fit(generator=gen,
                                                   steps_per_epoch=self.steps_per_epoch,
                                                   epochs=epochs,
                                                   validation_data=valid_gen,
                                                   validation_steps=self.validation_steps)

        # del exp
        self.results_table.set(run_id=run_id, hparams=hparams, val_loss=lowest_val_loss, epochs=epochs_seen)

    def rebuild(self, path):
        """
        Rebuilds the repository and results from a directory e.g. after a crash
        Args:
            path:

        Returns:

        """

    def _get_experiment(self, run_id, hparams=None):
        """
        Combines former init and load methods
        Args:
            run_id: tuple of (run, id)
            hparams: dict of hyper-parameters
        Returns:
            the retrieved experiment object

        """
        new_model = self.model_function(hparams) if hparams else None
        return Experiment(path=self.repo_dir, name=run_id_to_str(run_id), model=new_model)
