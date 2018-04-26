Welcome to SHERPA!
==================

SHERPA is a Python library for hyperparameter tuning of deep neural networks.

Its goal is to provide a platform for the implementation of innovations in
hyperparameter search algorithms. The tutorials section shows how to use some
of the implemented algorithms.

Installation
============


Installation from wheel
-----------------------

Download a copy of the wheel file from the dist folder in
git@gitlab.ics.uci.edu:uci-igb/sherpa.git

Make sure you have the most updated version of pip

::

    pip install --upgrade pip

Install wheel package if needed

::

    pip install wheel

Go to the directory where you downloaded the wheel and install sherpa
from wheel

::

    pip install sherpa-0.0.0-py2.py3-none-any.whl

If you used the wheel to install Sherpa you don’t need to set your
python path.

Installation from gitlab
------------------------

Clone into ``/your/path/`` from GitLab:

::

    cd /your/path/
    git clone git@gitlab.ics.uci.edu:uci-igb/sherpa.git

Add SHERPA to Python-path in your profile:

::

    export PYTHONPATH=$PYTHONPATH:/your/path/sherpa/

Install dependencies:

::

    pip install pandas
    pip install pymongo
    pip install numpy
    pip install scipy
    pip install scikit-learn
    pip install flask
    pip install drmaa
    pip install enum34  # if on < Python 3.4

Mongo DB
--------

Training models in parallel with SHERPA requires MongoDB. If you are using
a cluster, chances are that it is already installed, so check for that. Otherwise
the  .. _installation guide for Linux: https://docs.mongodb.com/manual/administration/install-on-linux/
is straightforward. For MacOS, MongoDB can either be installed via Homebrew

::

    brew update
    brew install mongodb

or via the .. _instructions: https://docs.mongodb.com/manual/tutorial/install-mongodb-on-os-x/ .



Example
-------

You can run an example to verify SHERPA is working:

::

    cd /your/path/sherpa/examples/
    python api_mode.py

And to verify SHERPA with MongoDB is working:

::

    cd /your/path/sherpa/examples/
    python runner_mode.pyFrom Keras to Sherpa in 30 seconds
==================================

Here we will show how to adapt a minimal Keras script so it can
be used with Sherpa. As starting point we use the "getting started in 30 seconds"
tutorial from the Keras webpage.

To run SHERPA you need a trial-script and a
runner-script. The first specifies the machine learning model and
will probably be very similar to the one you already have for Keras.
The second one will specify information about SHERPA and the optimization.

Trial-script
------------

For the ``trial.py`` we start by importing SHERPA and obtaining a trial. The
trial will contain the hyperparameters that we are tuning.

::

    import sherpa
    client = sherpa.Client()
    trial = client.get_trial()


Now we define the model, but for each tuning parameter we use
``trial.parameters[<name-of-parameter>]``. For example the number of
hidden units.

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
    model = Sequential()
    model.add(Dense(units=trial.parameters['num_units'], activation='relu', input_dim=100))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

For the training of the model, we include a
callback to send the information back to SHERPA at the end of each epoch
so it can update the state of it and decide if it should continue training.
Here you can include all the usual Keras callbacks as well.

Before:

::

    model.fit(x_train, y_train, epochs=5, batch_size=32)

After:

::

    callbacks = [client.keras_send_metrics(trial, objective_name='val_loss',
                 context_names=['val_acc'])]
    model.fit(x_train, y_train, epochs=5, batch_size=32, callbacks=callbacks)

Runner-script
-------------

Now we are going to create the runner-script and specify our hyperparameter
``num_units`` along with information for the hyperparameter algorithm, in this
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

And that's it! Now to run your hyperparameter optimization you just have to do:

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

Parameters: List of Parameter objects (sherpa.core.parameter).

Results: Dataframe storing the results of past trials.

Lower_is_better: Specifies if lower is better in performance metric of trials.

With this information you are free do select the new hyper-parameters in any way 
you want.

::

    import sherpa
    class MyAlgorithm(sherpa.algorithms.Algorithm):
        def get_suggestion(self, parameters, results, lower_is_better):
            # your code here
            return params_values_for_next_trial

For example let's create a genetic-like algorithm which takes the trials from the top 1/3 of the 
trials and combines them to create the new set of hyper-parameters. It will also 
randomly introduce a mutation 1/3 of the time.

The function get_candidate() will get the hyper-parameters
of a random trial among the top 1/3 and if there are very few trials, then it will generate them
randomly. Get_suggestion() is where the values for the hyper-parameters of the new trial will be decided. 

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


