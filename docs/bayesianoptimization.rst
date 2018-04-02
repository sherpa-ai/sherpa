Bayesian Optimization
=====================

Background
----------

Bayesian optimization for hyperparameter tuning as described for example
by Snoek et al. 2012 uses a flexible model to map from hyperparameter
space to objective values. In many cases this model is a Gaussian
Process (GP). Given inputs of hyperparameter configurations and outputs
of objective values the GP can be fitted. This GP can then be used to
make predictions about new hyperparameter configurations. Each
prediction can then be evaluated with respect to its utility via an
acquisiton function. The algorithm therefore consists of fitting the GP,
finding the maximum of the acquisition function, evaluating the chosen
hyperparameter configuration, and repeating the process. This yields an
optimization method that under certain conditions can be proved to be
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
the GP as justified by Duvenaud’s Kernel Cookbook.

    Snoek, Jasper, Hugo Larochelle, and Ryan P. Adams. “Practical
    bayesian optimization of machine learning algorithms.” Advances in
    neural information processing systems. 2012.

..

    https://www.cs.toronto.edu/~duvenaud/cookbook/
