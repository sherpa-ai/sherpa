API
===

Parameters
----------

.. autoclass:: sherpa.core.Continuous
   :noindex:

.. autoclass:: sherpa.core.Discrete
   :noindex:

.. autoclass:: sherpa.core.Choice
   :noindex:

.. autoclass:: sherpa.core.Ordinal
   :noindex:

Algorithms
----------

.. autoclass:: sherpa.algorithms.RandomSearch
   :noindex:

.. autoclass:: sherpa.algorithms.GridSearch
  :noindex:

Schedulers
----------

.. autoclass:: sherpa.schedulers.SGEScheduler
   :noindex:

.. autoclass:: sherpa.schedulers.LocalScheduler
  :noindex:

Running the Optimization
------------------------

.. autofunction:: sherpa.core.optimize
  :noindex:



Study
-----

A SHERPA optimization can also be run manually, without using
:code:`sherpa.optimize`. 

.. autoclass:: sherpa.core.Study
   :noindex:
   :members:
