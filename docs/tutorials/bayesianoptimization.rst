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

SHERPA Implementation
---------------------

SHERPA implements Bayesian optimization using a GP and the Expected
Improvement acquisition function. SHERPA uses a fixed length scale for the GP kernel
for the first points. After that the length scale is optimized with respect to
the marginal likelihood. The acquisition function is maximized by evaluating a
10000 random samples and numerically optimizing the continuous and
discrete parameters of the 50 best ones via *L-BFGS*.

SHERPA’s ``Discrete`` parameter is treated like a
continuous variable that is discretized after a value is suggested.
``Choice`` parameters are treated as categorical/one-hot variables in
the GP as justified by Duvenaud’s Kernel Cookbook. All parameters are scaled to
``[0, 1]`` internally and transformed back for the user.

The ``BayesianOptimization``
algorithm takes as argument an optional maximum number of trials and the number
of grid search points to initialize the optimization on.

.. autoclass:: sherpa.algorithms.BayesianOptimization
  :noindex:


Example
-------

Using Bayesian Optimization in SHERPA is straight forward. The parameter ranges
are defined as usual:

::

    parameters = [sherpa.Continuous('lrinit', [0.1, 0.01], 'log'),
                  sherpa.Continuous('momentum', [0., 0.99]),
                  sherpa.Continuous('lrdecay', [1e-2, 1e-7], 'log'),
                  sherpa.Continuous('dropout', [0., 0.5])]

When defining the algorithm the ``BayesianOptimization`` class is used:

::

    algorithm = sherpa.algorithms.BayesianOptimization(num_grid_points=2,
                                                       max_num_trials=150)

where ``num_grid_points`` describes the number of grid-search hyperparameter
configurations at the beginning of the optimization. Such seed configurations are required
so that the Bayesian Optimization model has data-points to make predictions. By
picking these configurations off a grid we make it easier for the space to be
explore subsequently. The ``num_grid_points`` applieds to continuous and discrete
parameters here. This means that ``num_grid_points=2`` implies ``d^2=4^2=16``
grid search configurations.


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

A full example for MNIST can be found in ``examples/mnistmlp/`` from the SHERPA
root. It can be run as:

::

    cd sherpa/examples/mnistmlp/
    python runner.py --algorithm BayesianOptimization


Below are the results for one run of this:

.. figure:: bayesopt-dashboard.jpg
   :alt: Dashboard after running Bayesian Optimization

The :ref:`LocalSearch tutorial <local-search>` will build on this result and
show how this can be refined and validated.


    Snoek, Jasper, Hugo Larochelle, and Ryan P. Adams. “Practical
    bayesian optimization of machine learning algorithms.” Advances in
    neural information processing systems. 2012.

..

    https://www.cs.toronto.edu/~duvenaud/cookbook/