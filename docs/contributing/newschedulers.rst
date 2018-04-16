.. _writing-schedulers:

Writing Schedulers
==================

A new scheduler inherits from the ``sherpa.schedulers.Scheduler`` class and
re-implements its methods ``submit_job``, ``get_status``, and ``kill_job``.

.. autoclass:: sherpa.schedulers.Scheduler
   :noindex:
   :members:
