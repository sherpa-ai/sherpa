SGE
===


Setting up your environment
---------------------------

In order to use SHERPA with a grid scheduler you will have to set up a profile
with environment variables. This will be loaded every time a job is submitted.
an SGE job will not load
your ``.bashrc`` so all necessary settings need to be in your profile.

Add MongoDB, DRMAA and SGE to your profile:

::

    module load mongodb/2.6
    export DRMAA_LIBRARY_PATH=/opt/sge/lib/lx-amd64/libdrmaa.so
    module load sge

Environment
-----------

You should have an environment-profile that sets path variables and
potentially loads a Python Virtual environment. All variable settings
above should go into that profile. Note that

SGE
---

SGE requires submit options. In Sherpa, those are defined as a string
via the ``submit_options`` argument in the scheduler. To run jobs on the
Arcus machines, typical submit options would be:
``-N myScript -P arcus.p -q arcus_gpu.q -l hostname='(arcus-1|arcus-2|arcus-3)'``.
The ``-N`` option defines the name. The SHERPA runner script can run
from any Arcus machine.