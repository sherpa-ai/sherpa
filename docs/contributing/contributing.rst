Development
===========

How to contribute
-----------------

The easiest way to contribute to SHERPA is to implement :ref:`new algorithms <writing-algorithms>` or
:ref:`new schedulers <writing-schedulers>`.


Style Guide
~~~~~~~~~~~

SHERPA uses Google style Python doc-strings (e.g. `here <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_ ).


Unit Testing
~~~~~~~~~~~~

Unit tests are organized in three scripts under ``/tests/`` from the SHERPA
root: ``test_sherpa.py`` tests core features of SHERPA, ``test_algorithms.py``
tests implemented algorithms, and ``test_schedulers.py`` tests schedulers. The
file ``long_tests.py`` does high level testing of SHERPA and takes longer to run.
All testing makes use of ``pytest``, especially ``pytest.fixtures``. The ``mock``
module is also used.


SHERPA Code Structure
---------------------

Study and Trials
~~~~~~~~~~~~~~~~

In Sherpa a parameter configuration corresponds to a ``Trial`` object and a
parameter optimization corresponds to a ``Study`` object. A trial has an ID
attribute and a ``dict`` of parameter name-value pairs.

.. autoclass:: sherpa.core.Trial
   :noindex:
   :members:

A study comprises the results of a number of trials. It also provides methods
for adding a new observation for a trial to the study (``add_observation``),
finalizing a trial (``finalize``), getting a new trial (``get_suggestion``),
and deciding whether a trial is performing worse than other trials and
should be stopped (``should_trial_stop``).

.. autoclass:: sherpa.core.Study
   :noindex:

In order to propose new trials or decide whether a trial should stop, the
study holds an ``sherpa.algorithms.Algorithm`` instance that yields new trials
and a ``sherpa.algorithms.StoppingRule`` that yields decisions about
performance. When using Sherpa in API-mode the user directly interacts with the study.

Runner
~~~~~~

The ``_Runner`` class automates the process of interacting with the study. It
consists of a loop that updates results, updates currently running jobs,
stops trials if necessary and submits new trials if necessary. In order to
achieve this it interacts with a ``sherpa.database._Database`` object and a
``sherpa.schedulers.Scheduler`` object.

.. autoclass:: sherpa.core._Runner
   :noindex:
