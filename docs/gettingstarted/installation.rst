

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
    pip install numpy
    pip install scipy
    pip install scikit-learn
    pip install flask
    pip install enum34  # if on < Python 3.4

You can run an example to verify SHERPA is working:

::

    cd /your/path/sherpa/examples/
    python api_mode.py

Note that to run hyperparameter optimizations in parallel with SHERPA requires
the installation of Mongo DB. Further instructions can be found in the Parallel
 Installation section of the documentation.