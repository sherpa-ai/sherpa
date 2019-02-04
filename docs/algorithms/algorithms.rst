.. _available-algorithms:

Available Algorithms
====================

This section provides an overview of the available hyperparameter optimization
algorithms in Sherpa. Below is a table that discusses use cases for each
algorithm. This is followed by a short comparison benchmark and the algorithms themselves.

+-------------------+-----------------------------------------------------------+
|                   | Use cases                                                 |
+-------------------+-----------------------------------------------------------+
| Grid Search       | | Great for understanding the                             |
|                   | | impact of one or two parameters.                        |
+-------------------+-----------------------------------------------------------+
| Random Search     | | More efficient than grid search when used with many     |
|                   | | hyperparameters. Great for getting                      |
|                   | | a full picture of the impact of many hyperparameters    |
|                   | | since hyperparameters are uniformly sampled from the    |
|                   | | whole space.                                            |
+-------------------+-----------------------------------------------------------+
| Local Search      | | Can quickly explore "tweaks" to a model that is already |
|                   | | good while using less trials than Random search or      |
|                   | | Bayesian optimization.                                  |
+-------------------+-----------------------------------------------------------+
| | GPyOpt Bayesian | | More efficient than Random search when the number of    |
| | Optimization    | | trials is sufficiently large.                           |
+-------------------+-----------------------------------------------------------+
| | Population      | | Can discover _schedules_ of training parameters and is  |
| | Based           | | therefore especially good for learning rate, momentum,  |
| | Training        | | batch size, etc.                                        |
+-------------------+-----------------------------------------------------------+

Comparison on MNIST MLP
~~~~~~~~~~~~~~~~~~~~~~~

The two figures below show five runs of Random Search against the same number of
Bayesian Optimization, and Local Search runs. The first figure shows the trial
index against the best validation loss achieved so far. The second figure shows
the mean across the five runs as a solid line and the minimum and maximum as
the shaded areas. Note that the Local Search finishes when no perturbation
yields an improvement which can be after a varying number of trials. It is
therefore not included in the mean figure.

.. figure:: individual-loss.png
   :alt: Individual Losses.

.. figure:: mean-loss.png
   :alt: Individual Losses.


Grid Search
~~~~~~~~~~~

.. autoclass:: sherpa.algorithms.GridSearch
  :noindex:


Random Search
~~~~~~~~~~~~~

.. autoclass:: sherpa.algorithms.RandomSearch
  :noindex:


Local Search
~~~~~~~~~~~~

.. autoclass:: sherpa.algorithms.LocalSearch
  :noindex:


Bayesian Optimization with GPyOpt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sherpa.algorithms.bayesian_optimization.GPyOpt
  :noindex:


Population Based Training
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sherpa.algorithms.PopulationBasedTraining
  :noindex:


