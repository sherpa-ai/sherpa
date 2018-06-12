SHERPA: A Python Hyperparameter Optimization Library
====================================================

.. figure:: https://docs.google.com/drawings/d/e/2PACX-1vRaTP5d5WqT4KY4V57niI4wFDkz0098zHTRzZ9n7SzzFtdN5akBd75HchBnhYI-GPv_AYH1zYa0O2_0/pub?w=522&h=150
    :figwidth: 100%
    :align: right
    :height: 150px
    :alt: SHERPA logo



SHERPA is a Python library for hyperparameter tuning of machine learning models. It provides:

* hyperparameter optimization for machine learning researchers
* a choice of hyperparameter optimization algorithms
* parallel computation that can be fitted to the user's needs
* a live dashboard for the exploratory analysis of results.

The documentation at http://parameter-sherpa.readthedocs.io/ provides installation instructions for parallel hyperparameter
optimizations and using the different optimization algorithms. See below for
a short example on what working with SHERPA looks like.


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


Installation from GitHub
========================

Clone from GitHub:

::

    git clone https://github.com/LarsHH/sherpa.git
    export PYTHONPATH=$PYTHONPATH:`pwd`/sherpa

Install dependencies:

::

    pip install pandas
    pip install numpy
    pip install scipy
    pip install scikit-learn
    pip install flask
    pip install enum34  # if on < Python 3.4

You can run an example to verify SHERPA is working:

::

    cd sherpa/examples/
    python simple.py

Note that to run hyperparameter optimizations in parallel with SHERPA requires
the installation of Mongo DB. Further instructions can be found in the
Parallel Installation section of the documentation.

