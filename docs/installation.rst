Installation
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