### Hyperparameter

```python
hobbit.core.Hyperparameter(name, distr_args, distribution='uniform')
```


A Hyperparameter instance captures the information about each of the hyperparameters to be optimized.

__Arguments__

- __name__: name of the hyperparameter. This will be used in the Model creation
- __distr_args__: List, Tuple or Dictionary, it used by the distribution function as arguments. 
		In the default case of a Uniform distribution these refer to the minimum and maximum values from 
		which to sample from. In general these are the  arguments taken as input by the corresponding numpy 
		distribution function.
- __distribution__: String, name of the distribution to be used for sampling the values. Must be numpy.random compatible. 
		  Uniform distribution is used as default.

__Examples__

```python
Hyperparameter('learning_rate', distr_args=(0.0001, 0.1), distribution='log-uniform'),
Hyperparameter('learning_rate', distr_args={low: 0.0001, high: 0.1}, distribution='uniform'),
Hyperparameter('activation', distr_args=[('sigmoid', 'tanh', 'relu')], distribution='choice')
```

