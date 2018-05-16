.. _keras-to-sherpa:

From Keras to Sherpa in 30 seconds
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

Now we are going to create the runner-script in a file called ``runner.py`` and
specify our hyperparameter ``num_units`` along with information for the
hyperparameter algorithm, in this case Random Search.

::

    import sherpa
    parameters = [sherpa.Choice('num_units', [100, 200, 300]),]
    alg = sherpa.algorithms.RandomSearch(max_num_trials=150)
    rval = sherpa.optimize(parameters=parameters,
                           algorithm=alg,
                           lower_is_better=True,  # Minimize objective
                           filename='./trial.py', # Python script to run, where the model was defined
                           scheduler=sherpa.schedulers.LocalScheduler(), # Run on local machine
                           )

And that's it! Now to run your hyperparameter optimization you just have to do:

::

    python runner.py


