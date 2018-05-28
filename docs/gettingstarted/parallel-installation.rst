Setup for Parallel Computation
==============================

Install dependencies:

::

    pip install pymongo

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

