from ourlibrary.job_schedulers import LocalJobScheduler
from ourlibrary.algorithms import Hyperband
from ourlibrary import Hyperparameter

def my_model(hparams):
	'''Keras model defintion returns compiled Keras model based on hparams'''
	return keras_model

my_dataset = load_my_dataset()

my_hparam_ranges = [Hyperparameter('learning_rate', 'log-uniform', (0.0001, 0.01)),
					Hyperparameter('activation', 'choice', ('sigmoid', 'tanh', 'relu')),
					Hyperparameter('dropout', 'uniform', (0., 1.))]

my_scheduler = LocalJobScheduler(devices=['gpu0', 'gpu1', 'gpu2'])

hband = Hyperband(my_model, my_dataset, my_hparam_ranges, my_scheduler, repo_dir='./example/')

hband.run(max_iterations_per_run=100, halving_factor=0.5)