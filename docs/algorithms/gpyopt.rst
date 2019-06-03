.. _bayesian-optimization:

Bayesian Optimization
=====================

Background
----------

Bayesian optimization for hyperparameter tuning
uses a flexible model to map from hyperparameter
space to objective values. In many cases this model is a Gaussian
Process (GP) or a Random Forest. The model is fitted to inputs of hyperparameter configurations and outputs
of objective values. It is then used to
make predictions about candidate hyperparameter configurations. Each
candidate-prediction can be evaluated with respect to its utility via an
acquisiton function - trading off exploration and exploitation. The algorithm therefore consists of fitting the model,
finding the hyperparameter configuration that maximize the acquisition function,
evaluating that configuration, and repeating the process.

GPyOpt Wrapper
--------------

SHERPA implements Bayesian optimization via a wrapper for the popular Bayesian optimization library GPyOpt
( https://github.com/SheffieldML/GPyOpt/ ). The ``GPyOpt``
algorithm in SHERPA has a number of arguments that specify the Bayesian optimization in GPyOpt.
The argument ``max_concurrent`` refers to the batch
size that GPyOpt produces at each step and should be chosen equal to the number
of concurrent parallel trials. The algorithm also accepts seed configurations
via the ``initial_data_points`` argument. This would be parameter configurations
that you know to be reasonably good and that can be used as starting points
for the Bayesian optimization. For the full specification see below. Note that
as of right now `sherpa.algorithms.GPyOpt` does not accept `Discrete` variables
with the option `scale='log'`.

.. autoclass:: sherpa.algorithms.GPyOpt
  :noindex:


Example
-------

Using GPyOpt Bayesian Optimization in SHERPA is straight forward. The parameter ranges
are defined as usual, for example:

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

The optimization is set up as shown in the  :ref:`Guide <guide>`. For example

::

    for trial in study:
        model = init_model(train.parameters)
        for iteration in range(num_iterations):
            training_error = model.fit(epochs=1)
            validation_error = model.evaluate()
            study.add_observation(trial=trial,
                                  iteration=iteration,
                                  objective=validation_error,
                                  context={'training_error': training_error})
        study.finalize(trial)

A full example for MNIST can be found in ``examples/mnist_mlp.ipynb`` from the SHERPA
root.