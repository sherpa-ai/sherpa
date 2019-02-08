.. _bayesian-optimization:

Bayesian Optimization
=====================

Background
----------

Bayesian optimization for hyperparameter tuning
uses a flexible model to map from hyperparameter
space to objective values. In many cases this model is a Gaussian
Process (GP) or a Random Forest. This model is fitted to inputs of hyperparameter configurations and outputs
of objective values. The model is used to
make predictions about candidate hyperparameter configurations. Each
candidate-prediction can be evaluated with respect to its utility via an
acquisiton function. The algorithm therefore consists of fitting the model,
finding the hyperparameter configuration that maximize the acquisition function,
evaluating that configuration, and repeating the process.

GPyOpt Wrapper
--------------

SHERPA uses a wrapper for the Bayesian optimization library GPyOpt
( https://github.com/SheffieldML/GPyOpt/ ).

SHERPA’s ``Discrete`` parameter is treated like a
continuous variable that is discretized after a value is suggested.
``Choice`` parameters are treated as GPyOpt discrete variables.

The ``GPyOpt``
algorithm in SHERPA takes as arguments a number of GPyOpt arguments as well as
the maximum number of trials. The argument ``max_concurrent`` refers to batch
size that GPyOpt produces at each step and should be chosen equal to the number
of concurrent parallel trials. The algorithm also accepts seed configurations
via the ``initial_data_points`` argument.

.. autoclass:: sherpa.algorithms.bayesian_optimization.GPyOpt
  :noindex:


Example
-------

Using GPyOpt Bayesian Optimization in SHERPA is straight forward. The parameter ranges
are defined as usual:

::

    parameters = [sherpa.Continuous('lrinit', [0.1, 0.01], 'log'),
                  sherpa.Continuous('momentum', [0., 0.99]),
                  sherpa.Continuous('lrdecay', [1e-2, 1e-7], 'log'),
                  sherpa.Continuous('dropout', [0., 0.5])]

When defining the algorithm the ``GPyOpt`` class is used:

::

    algorithm = sherpa.algorithms.GPyOpt(max_num_trials=150)

The ``max_num_trials`` argument is optional and specifies the number of trials after
which the algorithm will finish. If not specified the algorithm will keep running
and has to be cancelled by the user.

The optimization is set up as usual. Assume we have a trial script ``trial_script.py``
in which we want to minimize the loss, then we run the optimization as

::

    sherpa.optimize(parameters=parameters,
                    algorithm=algorithm,
                    lower_is_better=True,
                    filename='trial_script.py')

A full example for MNIST can be found in ``mnist_mlp.ipynb`` from the SHERPA
root.