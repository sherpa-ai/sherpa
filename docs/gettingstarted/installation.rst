Installation
============

Installation from GitHub
------------------------

Clone into ``/your/path/`` from GitHub:

::

    cd /your/path/
    git clone git@gitlab.ics.uci.edu:uci-igb/sherpa.git

Add SHERPA to Python-path:

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

You can run an example to verify SHERPA is working:

::

    cd /your/path/sherpa/examples/
    python api_mode.py

Note that to run hyperparameter optimizations in parallel with SHERPA requires
the installation of Mongo DB. If that is not an option take a look at this
example of optimizing a Random Forest in ``examples/randomforest/breastcancer.py`` without parallel computing.
See the ``Using the SHERPA API`` tutorial in the documentation for more information on
how to run SHERPA without Mongo DB.

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

To verify SHERPA *with* MongoDB is working:

::

    cd /your/path/sherpa/examples/
    python runner_mode.py

