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
    python runner_mode.py