.. _installation:

Installation
============

Installation from PyPi
~~~~~~~~~~~~~~~~~~~~~~

This is the most straightforward way to install Sherpa.

::

    pip install parameter-sherpa

However, since the source
is regularly updated we **recommend to clone from GitHub** as described below.


Installation from GitHub
~~~~~~~~~~~~~~~~~~~~~~~~

Clone from GitHub:

::

    git clone https://github.com/sherpa-ai/sherpa.git
    export PYTHONPATH=$PYTHONPATH:`pwd`/sherpa

Here you might want to add `export PYTHONPATH=$PYTHONPATH:/home/packages/sherpa/` to your
`.bash_profile` or `.bash_rc` so you won't have to run that line every time you
re-open the terminal. Replace `/home/packages/sherpa/` with the absolute path to
 the Sherpa folder on your system.

Install dependencies:

::

    pip install pandas
    pip install numpy
    pip install scipy
    pip install scikit-learn
    pip install flask
    pip install gpyopt

You can run an example to verify SHERPA is working:

::

    cd sherpa/examples/
    python simple.py

Note that to run hyperparameter optimizations in parallel with SHERPA requires
the installation of Mongo DB. Further instructions can be found in the
Parallel Installation section of the documentation.


