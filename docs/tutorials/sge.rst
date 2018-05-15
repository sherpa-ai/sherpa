SGE
===

The ``SGEScheduler`` class allows SHERPA to run hyperparameter optimizations
via the `Sun Grid Engine`. This works just like you would use a grid. While
SHERPA is running it calls ``qsub`` with a temporary bash script that loads your
environment, sets any SHERPA specific environment variables, and runs your
trial-script.

Using the ``SGEScheduler``, optimizations can easily be scheduled to run a large
number of concurrent instances of the trial-script. Below is the ``SGEScheduler``
class. Keep reading for more information on the environment and submit options.


.. autoclass:: sherpa.schedulers.SGEScheduler
  :noindex:


Your environment profile
------------------------

In order to use SHERPA with a grid scheduler you will have to set up a profile
with environment variables. This will be loaded every time a job is submitted.
an SGE job will not load your ``.bashrc`` so all necessary settings need to be
in your profile.

For example, in the case of training machine learning models on a GPU, the
profile might contain environment variables relating to CUDA or activate a
container that contains the requirements.
If you installed SHERPA via Git, then
the profile also might have to add the SHERPA folder to the ``PYTHONPATH``.
Finally, your environment might load a virtual environment that contains your
personal Python packages.


SGE submit options
------------------

SGE requires submit options. In Sherpa, those are defined as a string
via the ``submit_options`` argument in the scheduler. The string is attached
after the ``qsub`` command that SHERPA issues. To figure out what submit options
are needed for your setup you might want to refer to the cluster documentation,
group-wiki, or system administrator. In general, you will need

* ``-N``: the job name
* ``-P``: the project name
* ``-q``: the queue name.


Running it
----------

Note that while SHERPA is running in your runner-script it will repeatedly
submit your trial-script to SGE using ``qsub``. It is preferable to run the
runner-script itself in an interactive session since it is useful to be able
to monitor the output as it is running.