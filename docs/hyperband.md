### Hyperband

```python
hobbit.algorithms.Hyperband(model_function, hparam_ranges, repo_dir='./hyperband_repository', loss='val_loss', dataset=None, generator_function=None, train_gen_args=None, steps_per_epoch=None, validation_data=None, valid_gen_args=None, validation_steps=None)
```


An Algorithm instance initializes the entire pipeline needed to run a
hyperparameter optimization. The run() method is used to start
the optimization.

__Arguments__

- __model_function__: a function that takes a dictionary of hyperparameters
	as its only argument and returns a compiled Keras model object with
	those hyperparameters
- __hparam_ranges__: a list of Hyperparameter objects
- __repo_dir__: the directory to store weights and results table in
- __loss__: which loss to optimize e.g. 'val_loss', 'val_mse' etc.
- __dataset__: a dataset of the form ((x_train, y_train), (x_valid, y_valid))
	where x_, y_ are NumPy arrays
- __generator_function__: alternatively to dataset, a generator function can
	be passed. This is a function that returns a generator, not a generator 
	itself.
- __train_gen_args__: arguments to be passed to generator_function when
	producing a training generator
- __steps_per_epoch__: number of batches for one epoch of training when
	using a generator
- __validation_data__: generator function for the validation data, not the generator
- __valid_gen_args__: arguments to be passed to generator_function when
	producing a validation generator
- __validation_steps__: number of batches for one epoch of validation when
	using a generator

__Methods__

Runs the algorithm with **R** maximum epochs per stage and cut factor
**eta** between stages.

__run__

Depends on each optimization algorithm. For Hyperband this is:
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


__Notes__

At the beginning of the optimization Hobbit prints the schedule in which
it will initiate new configurations and continue old ones and the total
number of epochs. For example:

```python

----------------------------------------------------------------------------------------------------
	run=1		run=2		run=3		run=4		run=5
	models	epochs	models	epochs	models	epochs	models	epochs	models	epochs
Init	81	1	34	3	15	9	8	27	5	81
Cont	27	3	11	9	5	27	2	81
Cont	9	9	3	27	1	81
Cont	3	27	1	81
Cont	1	81
----------------------------------------------------------------------------------------------------
Total epochs=1902
----------------------------------------------------------------------------------------------------
```
This means in the first run 81 models are initialized and trained for one
epoch, then the 27 best are continued for 3 epochs etc., in the next run
34 configurations are initialized and 11 continued etc. The training
schedule depends on **R** and **eta**. You can experiment with the values
of those by calling the function ```visualize_hyperband_params()``` in
```hobbit.utils.monitoring_utils```. After the first model has been trained
Hobbit will also print an estimate of how long the entire optimization
will take.

__Reducing Optimization Time__

If you want to explore many models but don't have the time for a long
optimization we recommend to cut the number of training batches and
validation batches and have your generators start from a random point
in the dataset. That way an epoch is shorter in time, yet if you train
for multiple epochs you still get to use your entire dataset.


