.. _writing-algorithms:

Writing Your Own Algorithm
==========================

Now we will take a look at how to create a new algorithm which will
define the hyperparameters we will use to train the models. It defines
the hyperparameters to use in the trials. It does not define the algorithm
to train the model used in the trial, e.g. Stochasting Gradient Descent or Adam.

Every new algorithm inherits from the Algorithm Class and the main function we
need to define is ``get_suggestion()``. This function will receive information about
the parameters it needs to define and returns a dictionary of hyperparameter values
needed to train the next trial. The function ``get_suggestion()`` receives:

* ``parameters``: List of :ref:`Parameter objects <parameters-api>`.
* ``results``: Dataframe storing the results of past trials.
* ``lower_is_better``: Boolean specifying if lower is better in performance metric of trials.

With this information you are free to select the new hyperparameters in any way
you want.

::

    import sherpa
    class MyAlgorithm(sherpa.algorithms.Algorithm):
        def get_suggestion(self, parameters, results, lower_is_better):
            # your code here
            return params_values_for_next_trial

For example let's create a genetic-like algorithm which takes the trials from the top 1/3 of the
trials and combines them to create the new set of hyperparameters. It will also
randomly introduce a mutation 1/3 of the time.

The function ``get_candidate()`` will get the hyperparameters
of a random trial among the top 1/3 and if there are very few trials, then it will generate them
randomly. ``get_suggestion()`` is where the values for the hyperparameters of the new trial will be decided.

::

    import sherpa
    import numpy as np
    class MyAlgorithm(sherpa.algorithms.Algorithm):
        def get_suggestion(self, parameters, results, lower_is_better):
            """
            Create a new parameter value as a random mixture of some of the best
            trials and sampling from original distribution.

            Returns:
                dict: parameter values dictionary
            """
            # Choose 2 of the top trials and get their parameter values
            trial_1_params = self._get_candidate(parameters, results, lower_is_better)
            trial_2_params = self._get_candidate(parameters, results, lower_is_better)
            params_values_for_next_trial = {}
            for param_name in trial_1_params.keys():
                param_origin = np.random.randint(3)  # randomly choose where to get the value from
                if param_origin == 1:
                    params_values_for_next_trial[param_name] = trial_1_params[param_name]
                elif param_origin == 2:
                    params_values_for_next_trial[param_name] = trial_2_params[param_name]
                else:
                    for parameter_object in parameters:
                        if param_name == parameter_object.name:
                            params_values_for_next_trial[param_name] = parameter_object.sample()
            return params_values_for_next_trial

        def _get_candidate(self, parameters, results, lower_is_better, min_candidates=10):
            """
            Samples candidates parameters from the top 33% of population.

            Returns:
                dict: parameter dictionary.
            """
            if results.shape[0] > 0: # In case this is the first trial
                population = results.loc[results['Status'] != 'INTERMEDIATE', :]  # select only completed trials
            else: # In case this is the first trial
                population = None
            if population is None or population.shape[0] < min_candidates: # Generate random values
                for parameter_object in parameters:
                    trial_param_values[parameter_object.name] = parameter_object.sample()
                        return trial_param_values
            population = population.sort_values(by='Objective', ascending=lower_is_better)
            idx = numpy.random.randint(low=0, high=population.shape[0]//3)  # pick randomly among top 33%
            trial_all_values = population.iloc[idx].to_dict()  # extract the trial values on results table
            trial_param_values = {param.name: d[param.name] for param in parameters} # Select only parameter values
            return trial_param_values


