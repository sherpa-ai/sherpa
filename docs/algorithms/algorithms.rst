.. _available-algorithms:

Available Algorithms
====================

This section provides an overview of the available hyperparameter optimization
algorithms in Sherpa. Below is a table that discusses use cases for each
algorithm. This is followed by a short comparison benchmark and the algorithms themselves.

+-------------------+-----------------------------------------------------------+
|                   | Use cases                                                 |
+-------------------+-----------------------------------------------------------+
| | Grid Search     | | Great for understanding the                             |
|                   | | impact of one or two parameters.                        |
+-------------------+-----------------------------------------------------------+
| | Random Search   | | More efficient than grid search when used with many     |
|                   | | hyperparameters. Great for getting                      |
|                   | | a full picture of the impact of many hyperparameters    |
|                   | | since hyperparameters are uniformly sampled from the    |
|                   | | whole space.                                            |
+-------------------+-----------------------------------------------------------+
| | GPyOpt Bayesian | | More efficient than Random search when the number of    |
| | Optimization    | | trials is sufficiently large.                           |
+-------------------+-----------------------------------------------------------+
| | Asynchronous    | | Due to its early stopping, especially useful when it    |
| | Successive      | | would otherwise be infeasible to run a hyperparameter   |
| | Halving         | | optimization because of the computational cost.         |
+-------------------+-----------------------------------------------------------+
| | Local Search    | | Can quickly explore "tweaks" to a model that is         |
|                   | | already good while using less trials than Random search |
|                   | | or Bayesian optimization.                               |
+-------------------+-----------------------------------------------------------+
| | Population      | | Can discover *schedules* of training parameters and is  |
| | Based           | | therefore especially good for learning rate, momentum,  |
| | Training        | | batch size, etc.                                        |
+-------------------+-----------------------------------------------------------+

For the specification of each algorithm see below.

Comparison on MNIST MLP
~~~~~~~~~~~~~~~~~~~~~~~

The figure below shows the mean, minimum, and maximum across five runs of Random Search against the same number of
GPyOpt Bayesian optimization, and Local Search runs. For the Local Search the
individual trials are shown since each run finished after a different number of
trials.

.. figure:: mean-loss.png
   :alt: Individual Losses.

The currently available algorithms in Sherpa are listed below:

Grid Search
~~~~~~~~~~~

.. autoclass:: sherpa.algorithms.GridSearch
  :noindex:


Random Search
~~~~~~~~~~~~~

.. autoclass:: sherpa.algorithms.RandomSearch
  :noindex:


Bayesian Optimization with GPyOpt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sherpa.algorithms.GPyOpt
  :noindex:


Asynchronous Successive Halving aka Hyperband
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sherpa.algorithms.SuccessiveHalving
  :noindex:


Local Search
~~~~~~~~~~~~

.. autoclass:: sherpa.algorithms.LocalSearch
  :noindex:


Population Based Training
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: sherpa.algorithms.PopulationBasedTraining
  :noindex:

Repeat
~~~~~~

.. autoclass:: sherpa.algorithms.Repeat
  :noindex:

Iterate
~~~~~~~

.. autoclass:: sherpa.algorithms.Iterate
  :noindex: