Welcome to SHERPA!
==================

SHERPA is a Python library for hyperparameter tuning of deep neural networks.

Its goal is to provide a platform for the implementation of innovations in
hyperparameter search algorithms. The tutorials section shows how to use some
of the implemented algorithms.Installation
============

Setting up your environment
---------------------------

Add MongoDB, DRMAA and SGE to your profile:

::

    module load mongodb/2.6
    export DRMAA_LIBRARY_PATH=/opt/sge/lib/lx-amd64/libdrmaa.so
    module load sge

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

Add SHERPA and GPU_LOCK to Python-path in your profile:

::

    export PYTHONPATH=$PYTHONPATH:/your/path/sherpa/
    export PYTHONPATH=$PYTHONPATH:/extra/pjsadows0/libs/shared/gpu_lock/

Install dependencies:

::

    cd /your/path/sherpa
    pip install -e .

or

::

    pip install pandas
    pip install pymongo
    pip install numpy
    pip install scipy
    pip install scikit-learn
    pip install flask
    pip install drmaa
    pip install enum34  # if on < Python 3.4

Environment
-----------

You should have an environment-profile that sets path variables and
potentially loads a Python Virtual environment. All variable settings
above should go into that profile. Note that an SGE job will not load
your ``.bashrc`` so all necessary settings need to be in your profile.

SGE
---

SGE requires submit options. In Sherpa, those are defined as a string
via the ``submit_options`` argument in the scheduler. To run jobs on the
Arcus machines, typical submit options would be:
``-N myScript -P arcus.p -q arcus_gpu.q -l hostname='(arcus-1|arcus-2|arcus-3)'``.
The ``-N`` option defines the name. The SHERPA runner script can run
from any Arcus machine.

Example
-------

You can run an example by doing:

::

    cd /your/path/sherpa/examples/bianchini/
    python runner.py --env <path/to/your/environment>

From Keras to Sherpa in 30 seconds
==================================

Here we will show how to adapt a minimal Keras script so it can
be used with Sherpa. As starting point we use the "getting started in 30 seconds"
tutorial from the Keras webpage.

As mentioned in the previous section you need a trial-script and a
runner-script. The first specifies the machine learning model and
will probably be very similar to the one you already have for Keras.
The second one will specify information about Sherpa and the optimization.

Trial-script
------------

For the ``trial.py`` we need a ``define_model()`` function which initializes
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

Next we will get the information about the parameters from SHERPA and
specify how to train the model in this specific trial. We include a
callback to send the information back to SHERPA at the end of each epoch
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


