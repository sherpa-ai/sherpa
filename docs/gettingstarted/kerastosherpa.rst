.. _keras-to-sherpa-api:

From Keras to Sherpa in 30 seconds
==================================

This example will show how to adapt a minimal Keras script so it can
be used with SHERPA. As starting point we use the "getting started in 30 seconds"
tutorial from the Keras webpage.

We start out with this piece of Keras code:

::

    from keras.models import Sequential
    from keras.layers import Dense
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=100))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

We want to tune the number of hidden units via Random Search. To do that, we
define one parameter of type `Discrete`.
We also use the `BayesianOptimization` algorithm with maximum number of trials 50.

::

    import sherpa
    parameters = [sherpa.Discrete('num_units', [50, 200])]
    alg = sherpa.algorithms.BayesianOptimization(max_num_trials=50)

We use these objects to create a SHERPA Study:

::

    study = sherpa.Study(parameters=parameters,
                         algorithm=alg,
                         lower_is_better=True)

We obtain `trials` by iterating over the study. Each `trial` has a `parameter`
attribute that contains the ``num_units`` parameter value. We can use that value
to create our model.

::

    for trial in study:
        model = Sequential()
        model.add(Dense(units=trial.parameters['num_units'],
                        activation='relu', input_dim=100))
        model.add(Dense(units=10, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=5, batch_size=32,
                  callbacks=[study.keras_callback(trial, objective_name='val_loss')])
        study.finalize(trial)

During training, objective values will be added to the SHERPA study via the
callback. At the end of training ``study.finalize`` completes this trial. This means
that no more observation will be added to this trial.

When the ``Study`` is created, SHERPA will display the dashboard address. If you
put the address into your browser you will see the dashboard as shown below. As a next step you
can take a look at this example of optimizing a Random Forest in
``sherpa/examples/randomforest.py``.

.. figure:: https://drive.google.com/uc?export=view&id=1G85sfwLicsQKd3-1xN7DZowQ0gHAvzGx
   :alt: SHERPA Dashboard.

