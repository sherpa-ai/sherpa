.. SHERPA documentation master file, created by
    sphinx-quickstart on Thu Mar 29 12:22:17 2018.
    You can adapt this file completely to your liking, but it should at least
    contain the root `toctree` directive.

.. include:: ./welcome.rst

Documentation Contents
----------------------

.. toctree::
    :maxdepth: 1

    self

.. toctree::
    :caption: Getting Started
    :maxdepth: 1

    gettingstarted/installation
    gettingstarted/kerastosherpa
    gettingstarted/guide

.. toctree::
    :caption: Algorithms
    :maxdepth: 1

    algorithms/algorithms
    algorithms/gpyopt
    algorithms/keras_mnist_mlp_population_based_training
    algorithms/keras_mnist_mlp_successive_halving
    algorithms/localsearch
    algorithms/writingyourown

.. toctree::
    :caption: Parallel
    :maxdepth: 1

    parallel/kerastosherpa-parallel
    parallel/parallel-installation
    parallel/parallel-guide
    parallel/sge

.. toctree::
    :caption: API
    :maxdepth: 1

    api/coreapi
    api/schedulersapi
    api/algorithmsapi

.. toctree::
    :caption: Contributing
    :maxdepth: 1

    contributing/contributing
    contributing/newschedulers