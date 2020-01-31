.. _guide:

A Guide to SHERPA
=================


Parameters
~~~~~~~~~~

Hyperparameters are defined via ``sherpa.Parameter`` objects. Available are


- ``sherpa.Continuous``: Represents continuous parameters such as `weight-decay` multiplier. Can also be thought of as `float`.
- ``sherpa.Discrete``: Represents discrete parameters such as `number of hidden units` in a neural network. Can also be thought of as `int`.
- ``sherpa.Ordinal``: Represents categorical ordered parameters. For example `minibatch` size could be an ordinal parameter taking values `8`, `16`, `32`, etc. Can also be thought of as `list`.
- ``sherpa.Choice``: Represents unordered categorical parameters such as `activation` function in a neural network. Can also be thought of as a `set`.


Every parameter takes a ``name`` and ``range`` argument. The ``name`` argument
is simply the name of the hyperparameter. The ``range`` is either the lower and
upper bound of the range, or the possible values in the case of
``sherpa.Ordinal`` and ``sherpa.Choice``.
The ``sherpa.Continuous`` and ``sherpa.Discrete`` parameters also take a ``scale``
argument which can take values ``linear`` or ``log``. This describes whether
values are sampled uniformly on a linear or a log scale.

Hyperparameters are defined as a list to be passed to the ``sherpa.Study`` down
the line. For example:

::

    parameters = [sherpa.Continuous(name='lr', range=[0.005, 0.1], scale='log'),
                  sherpa.Continuous(name='dropout', range=[0., 0.4]),
                  sherpa.Ordinal(name='batch_size', range=[16, 32, 64]),
                  sherpa.Discrete(name='num_hidden_units', range=[100, 300]),
                  sherpa.Choice(name='activation', range=['relu', 'elu', 'prelu'])]

Note that it is generally recommended not to represent continuous or discrete
parameters as categorical. This is due to the fact that exploring a range of
values rather than discrete options yields much more information to understand
the relationship between the hyperparameter and the outcome.


The Algorithm
~~~~~~~~~~~~~

The algorithm refers to the procedure that determines how hyperparameter
configurations are chosen and in some cases the resource they are assigned.
All available algorithms can be found in ``sherpa.algorithms``. The description
in :ref:`Available Algorithms <available-algorithms>` gives an in-depth view
of what algorithms are available, their arguments, and when one might chose one algorithm over
another. The ``sherpa.algorithms`` module is also home to `stopping rules`.
Those are procedures that define if a trial should be stopped before its
completion.
The initialization of the algorithm is simple. For example:

::

    algorithm = sherpa.algorithms.RandomSearch(max_num_trials=150)


where ``max_num_trials`` stands for the number of trials after which the
algorithm will finish.


The Study
~~~~~~~~~

In Sherpa a `Study` represents the hyperparameter optimization itself. It holds
references to the parameter ranges, the algorithm, the results that have been
gathered, and provides an interface to obtain a new trial, or add results from
previously suggested trial. It also starts the dashboard in the background.
When initializing the study it expects references to the parameter ranges, the
algorithm, and at minimum a boolean variable on whether lower objective values
are better. For a full list of the arguments see the
:ref:`Study-API reference <study-api>`.

::

    study = sherpa.Study(parameters=parameters,
                     algorithm=algorithm,
                     lower_is_better=False)


In order to obtain a first trial one can either call ``Study.get_suggestion()``
or directly iterate over the ``Study`` object.

::

    # To get a single trial
    trial = study.get_suggestion()

    # Or directly iterate over the study
    for trial in study:
        ...

The ``Trial`` object has an ``id`` attribute and a ``parameters`` attribute.
The latter contains a dictionary with a hyperparameter configuration from the
previously specified ranges provides by the defined algorithm. The parameter
configuration can be used to initialize, train, and evaluate a model.

::

    model = init_model(train.parameters)

During training ``Study.add_observation`` can be used to add intermediate metric
values from the model training.

::

    for iteration in range(num_iterations):
        training_error = model.fit(epochs=1)
        validation_error = model.evaluate()
        study.add_observation(trial=trial,
                              iteration=iteration,
                              objective=validation_error,
                              context={'training_error': training_error})

Once the model has completed training Sherpa expects a call to the
``Study.finalize`` function.

::

    study.finalize(trial)

This can be put together in a double for-loop of the form:

::

    for trial in study:
        model = init_model(trial.parameters)
        for iteration in range(num_iterations):
            training_error = model.fit(epochs=1)
            validation_error = model.evaluate()
            study.add_observation(trial=trial,
                                  iteration=iteration,
                                  objective=validation_error,
                                  context={'training_error': training_error})
        study.finalize(trial)



Visualization
~~~~~~~~~~~~~

Once the ``Study`` object is initialized it will output the following:

::

    SHERPA Dashboard running on http://...

Following that link brings up the dashboard. The figure at the top of the dashboard
is a parallel coordinates plot. It allows the user to brush over axes and thereby
restrict ranges of the trials she wants to see. This is useful to find what
objective values correspond to hyperparameters of a certain range. Similarly,
one can brush over the objective value axis to find the best performing
configurations. The table in the bottom left of the dashboard is linked to the
plot. Therefore, it is easy to see what exact hyperparameters the filtered
trials correspond to. One can also sort the table by any of its columns. Lastly,
on the bottom right is a line plot that shows the progression of objective
values for each trial. This is useful in analyzing how and if the training
converges. Below is a screenshot of the dashboard towards the end of a study.


.. figure:: https://drive.google.com/uc?export=view&id=1G85sfwLicsQKd3-1xN7DZowQ0gHAvzGx
   :alt: SHERPA Dashboard.
