.. _local-search:

Local Search
============

Background
----------

The goal for the Local Search algorithm is to start with a good hyperparameter
configuration and test if it can be improved. The starting configuration could
have been obtained through one of the other algorithms or from hand-tuning. The
algorithm starts by evaluating the seed_configuration. It then perturbs one
parameter at a time. If a new configuration achieves a better objective value
than the seed then the new configuration is made the new seed.

Perturbations are applied as multiplication by a factor in the case of
``Continuous`` or ``Discrete`` variables. The default values are `0.8` and
`1.2`. These can be modified via the ``perturbation_factors`` argument. In the
case of ``Ordinal`` variables, the parameter is shifted one up or down in the
provided values. ``Choice`` variables are not allowed.

Example
-------

In this example we will work with the MNIST fully connected neural network from
the :ref:`Bayesian Optimization tutorial <bayesian-optimization>`. We had tuned
`initial learning rate`, `learning rate decay`, `momentum`, and `dropout` rate.
The top parameter configuration we obtained was:

* `initial learning rate`: `0.038`
* `learning rate decay`: `1.2e-4`
* `momentum`: `0.92`
* `dropout`: `0.`

rounded to two digits. We use this as ``seed_configuration`` in the Local Search.
We set the ``perturbation_factors`` as ``(0.9, 1.1)``. The algorithm will
multiply one parameter by `0.9` or `1.1` at a time and see if these local
changes can improve performance. If all changes have been tried and none improves
on the seed configuration the algorithm stops.

From the dashboard we can find the best results as:
