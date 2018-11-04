.. _installation:

Installation
============

Installation from PyPi
~~~~~~~~~~~~~~~~~~~~~~

This is the most straightforward way to install Sherpa. The source code may
however be slightly older than what is found on the GitHub.

::

    pip install parameter-sherpa


Installation from GitHub
~~~~~~~~~~~~~~~~~~~~~~~~

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


