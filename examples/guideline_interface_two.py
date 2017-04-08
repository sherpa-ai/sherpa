from ourlibrary.job_schedulers import RemoteJobScheduler
from ourlibrary.algorithms import Hyperband
from ourlibrary import Hyperparameter

def my_model(hparams):
	'''Keras model defintion returns compiled Keras model based on hparams'''
	return keras_model

my_generator = dataset_generator()

my_hparam_ranges = [Hyperparameter('learning_rate', 'log-uniform', (0.0001, 0.01)),
				 	Hyperparameter('activation', 'choice', ('sigmoid', 'tanh', 'relu'))
				 	Hyperparameter('dropout', 'uniform', (0., 1.)),
				 	Hyperparameter('num_units_per_layer', 'choice', tuple(range(100, 200)))]

my_scheduler = RemoteJobScheduler(devices=[('lhertel@arcus-4.ics.uci.edu', 'gpu0'),
										   ('lhertel@arcus-3.ics.uci.edu', 'gpu1'),
										   ('lhertel@arcus-7.ics.uci.edu', 'gpu3')]

hband = Hyperband(my_model, my_generator, my_hparam_ranges, my_scheduler, repo_dir='./example/')

hband.run(max_iterations_per_run=100, halving_factor=0.5)


#### Say we press Ctrl+C at some point... then we enter the above again but instead of hband.run()
hband.recreate(from_dir='./example'/)


#### Or we recreate everything from dir + config