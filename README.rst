SHERPA: A Python Hyperparameter Optimization Library
====================================================

.. figure:: https://docs.google.com/drawings/d/e/2PACX-1vRaTP5d5WqT4KY4V57niI4wFDkz0098zHTRzZ9n7SzzFtdN5akBd75HchBnhYI-GPv_AYH1zYa0O2_0/pub?w=522&h=150
    :figwidth: 100%
    :align: right
    :height: 150px
    :alt: SHERPA logo

.. image:: https://img.shields.io/badge/License-GPL%20v3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0

.. image:: https://pepy.tech/badge/parameter-sherpa
   :target: https://pepy.tech/project/parameter-sherpa


SHERPA is a Python library for hyperparameter tuning of machine learning models. It provides:

* hyperparameter optimization for machine learning researchers
* it can be used with any Python machine learning library such as Keras, Tensorflow, or Scikit-Learn
* a choice of hyperparameter optimization algorithms such as **Bayesian optimization via GPyOpt**, **Asynchronous Successive Halving** (aka Hyperband), and **Population Based Training**.
* **parallel** computation that can be fitted to the user's needs
* a live **dashboard** for the exploratory analysis of results.

Install via ``pip install parameter-sherpa``. The documentation at http://parameter-sherpa.readthedocs.io/ provides tutorials on using the different optimization algorithms and installation instructions for parallel hyperparameter
optimizations. Take a look at the demo
video by clicking on the image below or read on to find out more.

.. image:: http://img.youtube.com/vi/L95sasMLgP4/0.jpg
   :target: https://www.youtube.com/watch?feature=player_embedded&v=L95sasMLgP4

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
   

Installation from PyPi
======================

::

    pip install parameter-sherpa


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

