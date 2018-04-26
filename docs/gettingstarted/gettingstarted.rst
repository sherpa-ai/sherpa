Writing your Optimization
=========================

An optimization in SHERPA consists of a trial-script and a
runner-script.

Trial-script
------------

The trial-script trains your machine learning model with a given
parameter-configuration and sends metrics to SHERPA. To get a trial, use the ``Client``:

::

    import sherpa

    client = sherpa.Client()
    trial = client.get_trial()

The client will connect to the MongoDB instance created by the Runner-script (more below).
From that it obtains a hyperparameter configuration i.e. a trial.
The trial contains the parameter configuration for your training:

::

    # Model training
    num_iterations = 10
    for i in range(num_iterations):
        pseudo_objective = trial.parameters['param_a'] / float(i + 1) * trial.parameters['param_b']
        client.send_metrics(trial=trial, iteration=i+1,
                            objective=pseudo_objective)

During training ``send_metrics`` is used every iteration to return
objective values to SHERPA i.e. send them to the MongoDB instance. When using
Keras the client also has a callback ``Client.keras_send_metrics`` that can be
used directly.

Runner-script
-------------

The runner-script defines the optimization and runs SHERPA. Parameters
are defined as a list of Parameter-objects:

::

    import sherpa
    parameters = [sherpa.Choice(name="param_a",
                                range=[1, 2, 3]),
                  sherpa.Continuous(name="param_b",
                                    range=[0, 1])]

Once you decided on the parameters and their ranges you can choose an
optimization algorithm:

::

    algorithm = sherpa.algorithms.RandomSearch(max_num_trials=10)

Schedulers allow to run an optimization on one machine or a cluster:

::

    scheduler = sherpa.schedulers.LocalScheduler()

The optimization is run via:

::

    results = sherpa.optimize(parameters=parameters,
                              algorithm=algorithm,
                              lower_is_better=True,
                              filename=filename,
                              output_dir=tempdir,
                              scheduler=scheduler,
                              max_concurrent=2,
                              verbose=1)

The code for this example can be run as
``python ./examples/runner_mode.py`` from the SHERPA root.


