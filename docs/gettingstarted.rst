Getting Started
===============

An optimization in SHERPA consists of a trial-script and a
runner-script.

Trial-script
------------

The trial-script trains your machine learning model with a given
parameter-configuration and sends metrics to SHERPA. To get a trial:

::

    import sherpa

    client = sherpa.Client()
    trial = client.get_trial()

The trial contains the parameter configuration for your training:

::

    # Model training
    num_iterations = 10
    for i in range(num_iterations):
        pseudo_objective = trial.parameters['param_a'] / float(i + 1) * trial.parameters['param_b']
        client.send_metrics(trial=trial, iteration=i+1,
                            objective=pseudo_objective)
        # print("Trial {} Iteration {}.".format(trial.id, i+1))
    # print("Trial {} finished.".format(trial.id))

During training ``send_metrics`` is used every iteration to return
objective values to SHERPA

Runner-script
-------------

The runner-script defines the optimization and runs SHERPA. Parameters
are defined as a list:

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


From Keras to Sherpa in 30 seconds
===================================

Here we will show how to adapt a minimal Keras script so it can 
be used with Sherpa. As starting point we use the "getting started in 30 seconds"
tutorial from the Keras webpage.

As mentioned in the previous section you need a trial-script and a 
runner-script. The first specifies the machine learning model and  
will probably be very similar to the one you already have for Keras. 
The second one will specify information about Sherpa and the optimization.

Trial-script
--------------

For the trial.py we need a define_model() function which initializes 
and returns a compiled Keras model. The function receives a dictionary
of hyperparameters which can be used to specify how each model will be
different, for example the number of hidden units.

Before:

::

    from keras.models import Sequential
    from keras.layers import Dense
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=100))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

After:

::

    from keras.models import Sequential
    from keras.layers import Dense
    def define_model(params):
        model = Sequential()
        model.add(Dense(units=params('num_units'), activation='relu', input_dim=100))
        model.add(Dense(units=10, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
        return model

Next we will get the information about the parameters from sherpa and
specify how to train the model in this specific trial. We include a
callback to send the information back to Sherpa at the end of each epoch
so it can update the state of it and decide if it should continue training.
Here you can include all the usual Keras callbacks as well.

Before:

::

    model.fit(x_train, y_train, epochs=5, batch_size=32)

After:

:: 

    import sherpa
    client = sherpa.Client()
    trial = client.get_trial()
    model   = define_model(trial.parameters)
    send_call = lambda epoch, logs: client.send_metrics(trial=trial,
                                                        iteration=epoch,
                                                        objective=logs['val_acc'],
                                                        context={'val_loss': logs['val_loss']})
    callbacks = [keras.callbacks.LambdaCallback(on_epoch_end=send_call)]
    model.fit(x_train, y_train, epochs=5, batch_size=32, callbacks=callbacks)

Runner-script
--------------

Now we are going to create the runner-script and specify our hyperparameter
'num_units' along with information for the hyperparameter algorithm, in this
case Random Search.

::

    import sherpa
    parameters = [sherpa.Choice('num_units', [100, 200, 300]),]
    alg = sherpa.algorithms.RandomSearch(max_num_trials=150)
    rval = sherpa.optimize(parameters=parameters,
                           algorithm=alg,
                           lower_is_better=False,
                           filename='./trial.py', # Python script to run, where the model was defined
                           scheduler=sherpa.schedulers.LocalScheduler(), # Run on local machine
                           )
And that's it! Now to run your model you just have to do:

::

    python runner.py


Creating a new hyperparameter optimization algorithm
=====================================================

Now we will take a look at how to create a new algorithm which will
define the hyper-parameters we will use to train the models. It defines
the hyperparameters to use in the trials. It does not define the algorithm 
to train the model used in the trial, e.g. Stochasting Gradient Descent or Adam.

Every new algorithm inherits from the Algorithm Class and the main function we
need to define is get_suggestion(). This function will receive information about
the parameters it needs to define and returns a dictionary of hyperparameter values
needed to train the next trial. The function get_suggestion() receives:
- Parameters: List of Parameter objects (sherpa.core.parameter).
- Results: Dataframe storing the results of past trials.
- Lower_is_better: Specifies if lower is better in performance metric of trials.

With this information you are free do select the new hyper-parameters in any way
you want.

::
    import sherpa
    class MyAlgorithm(sherpa.algorithms.Algorithm):
        def get_suggestion(self, parameters, results, lower_is_better):
            # your code here
            return params_values_for_next_trial

For example let's create an algorithm which takes the trials from the top 1/3 of the
trials and combines them to create the new set of hyper-parameters.

::
    import sherpa
    import numpy as np
    class MyAlgorithm(sherpa.algorithms.Algorithm):
        def get_suggestion(self, parameters, results, lower_is_better):
            """
            Create a new parameter value as a random mixture of some of the best
            trials and sampling from original distribution.
            
            Returns
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
    
            Returns
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
        
