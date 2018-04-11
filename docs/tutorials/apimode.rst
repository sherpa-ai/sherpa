Using the SHERPA API
====================

Instead of running an optimization via :code:`sherpa.optimize` with a scheduler
and a database running in the background, an optimization can also be run
directly via the API. This is useful for machine learning models that train
quickly enough that parallel training is not required. The optimization is set
up as before, defining parameter ranges and an algorithm:

::

    import sherpa

    parameters = [sherpa.Choice(name="param_a",
                                range=[1, 2, 3]),
                  sherpa.Continuous(name="param_b",
                                    range=[0, 1])]

    algorithm = sherpa.algorithms.BayesianOptimization()

We then define a study and iterate over that study. Each iteration is a trial.
For each trial we manually add observations and finalize it once training is
finished.

::

    study = sherpa.Study(parameters=parameters,
                         algorithm=algorithm,
                         lower_is_better=True)

    num_iterations = 10

    # get trials from study by iterating or use study.get_suggestion()
    for trial in study:
        print("Trial {}:\t{}".format(trial.id, trial.parameters))

        # Simulate model training
        for i in range(num_iterations):

            # access parameters via trial.parameters and id via trial.id
            pseudo_objective = trial.parameters['param_a'] / float(i + 1) * trial.parameters['param_b']

            # add observations once or multiple times
            study.add_observation(trial=trial,
                                  iteration=i+1,
                                  objective=pseudo_objective)

            study.finalize(trial=trial)

The study will still run the dashboard, however no MongoDB or scheduler is being
run.