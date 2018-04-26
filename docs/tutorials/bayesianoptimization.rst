.. _bayesian-optimization:

Bayesian Optimization
=====================

Background
----------

Bayesian optimization for hyperparameter tuning (e.g. Snoek et al. 2012)
uses a flexible model to map from hyperparameter
space to objective values. In many cases this model is a Gaussian
Process (GP) or a Random Forest. This model is fitted to inputs of hyperparameter configurations and outputs
of objective values. The model is used to
make predictions about candidate hyperparameter configurations. Each
candidate-prediction can be evaluated with respect to its utility via an
acquisiton function. The algorithm therefore consists of fitting the model,
making predictions for many candidates,
finding the candidate that maximizes the acquisition function, evaluating the chosen
candidate hyperparameter configuration, and repeating the process. This yields an
optimization method that under certain conditions is proven to be
consistent.

SHERPA Implementation
---------------------

SHERPA implements Bayesian optimization using a GP and an Expected
Improvement acquisition function. In contrast to Snoek et al. 2012,
SHERPA obtains the hyperparameters of the Gaussian Process via
maximization of the marginal likelihood. At this point there is no
thorough treatment of parallel evaluation in place. However, in practice
this is not an issue since the optimium of the acquistion function is
found via random sampling and numerical optimization of the best
samples. It is therefore unlikely that the same configuration is
suggested twice. SHERPA’s ``Discrete`` parameter is treated like a
continuous variable that is discretized after a value is suggested.
``Choice`` parameters are treated as categorical/one-hot variables in
the GP as justified by Duvenaud’s Kernel Cookbook. The ``BayesianOptimization``
algorithm takes as argument an optional maximum number of trials.

.. autoclass:: sherpa.algorithms.BayesianOptimization
  :noindex:


Example
-------

Using Bayesian Optimization in SHERPA is straight forward. The parameter ranges
are defined as usual:

::

    parameters = [sherpa.Continuous('lrinit', [0.1, 0.01], 'log'),
                  sherpa.Continuous('momentum', [0., 0.99]),
                  sherpa.Continuous('lrdecay', [1e-2, 1e-7], 'log')]

When defining the algorithm the ``BayesianOptimization`` class is used:

::

    algorithm = sherpa.algorithms.BayesianOptimization(num_random_seeds=10,
                                                       max_num_trials=150)

where ``num_random_seeds`` describes the number of randomly sampled hyperparameter
configurations at the beginning of the optimization. Random samples are required
so that the Bayesian Optimization model has data-points to make predictions. The
``max_num_trials`` argument is optional and specifies the number of trials after
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