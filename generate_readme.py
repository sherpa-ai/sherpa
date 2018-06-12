import os

welcome_text = """SHERPA: A Python Hyperparameter Optimization Library
====================================================

.. figure:: https://docs.google.com/drawings/d/e/2PACX-1vRaTP5d5WqT4KY4V57niI4wFDkz0098zHTRzZ9n7SzzFtdN5akBd75HchBnhYI-GPv_AYH1zYa0O2_0/pub?w=522&h=150
    :figwidth: 100%
    :align: right
    :height: 150px
    :alt: SHERPA logo



SHERPA is a Python library for hyperparameter tuning of machine learning models. It provides:

* hyperparameter optimization for machine learning researchers
* a choice of hyperparameter optimization algorithms
* parallel computation that can be fitted to the user's needs
* a live dashboard for the exploratory analysis of results.

The documentation at http://parameter-sherpa.readthedocs.io/ provides installation instructions for parallel hyperparameter
optimizations and using the different optimization algorithms. See below for
a short example on what working with SHERPA looks like.

"""


filenames = ['gettingstarted/kerastosherpa.rst', 'gettingstarted/installation.rst']
with open('README.rst', 'w') as outfile:
    outfile.write(welcome_text)
    for fname in filenames:
        with open(os.path.join('docs', fname)) as infile:
            data = infile.read().splitlines(True)
        outfile.writelines(data[1:])
