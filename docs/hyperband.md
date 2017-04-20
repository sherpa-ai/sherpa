### Hyperband

```python
hobbit.algorithms.Hyperband(model_function, hparam_ranges, repo_dir='./hyperband_repository', dataset=None, generator_function=None, train_gen_args=None, steps_per_epoch=None, valid_gen_args=None, validation_steps=None)
```


A Hyperband instance initializes the entire pipeline needed to run a
Hyperband hyperparameter optimization. The run() method is used to start
the optimization.

__Arguments__

- __model_function__: a function that takes a dictionary of hyperparameters
	as its only argument and returns a compiled Keras model object with
	those hyperparameters
- __hparam_ranges__: a list of Hyperparameter objects
- __repo_dir__: the directory to store weights and results table in
- __dataset__: a dataset of the form ((x_train, y_train), (x_valid, y_valid))
	where x_, y_ are NumPy arrays
- __generator_function__: alternatively to dataset, a generator function can
	be passed or a tuple of generator functions. This is a function
	that returns a generator, not a generator itself. For a tuple
	the first item is the generator function for training, the second
	for validation.
- __train_gen_args__: arguments to be passed to generator_function when
	producing a training generator
- __steps_per_epoch__: number of batches for one epoch of training when
	using a generator
- __valid_gen_args__: arguments to be passed to generator_function when
	producing a validation generator
- __validation_steps__: number of batches for one epoch of validation when
	using a generator

__Methods__

Runs the algorithm with **R** maximum epochs per stage and cut factor
**eta** between stages.

__run__

- __R__: The maximum epochs per stage. Hyperband has multiple runs each of
	which goes through multiple stages to discard configurations. At each
	of those stages Hyperband will train for a total of R epochs
- __eta__: The cut-factor. After each stage Hyperband will reduce the number
	of configurations by this factor. The training
	iterations for configurations that move to the next stage increase
	by this factor

__Example__

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
