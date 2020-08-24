#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine
from __future__ import print_function
import io
import os
import sys
from shutil import rmtree
import argparse
from setuptools import find_packages, setup, Command


# parser = argparse.ArgumentParser()
# parser.add_argument('--parallel', help='Install packages required'
#                                        'by Parallel-mode.',
#                     action='store_true', default=False)
# args = parser.parse_args()

# Package meta-data.
NAME = 'parameter-sherpa'
DESCRIPTION = 'Hyperparameter Optimization for Machine Learning Models.'
URL = 'https://github.com/sherpa-ai/sherpa'
EMAIL = 'lhertel@uci.edu'
AUTHOR = 'Lars Hertel, Peter Sadowski, and Julian Collado'

REQUIRED = [
    'pandas>=0.20.3',
    'pymongo>=3.5.1',
    'numpy>=1.8.2',
    'scipy>=1.0.0,<=1.4.1',
    'scikit-learn>=0.19.1',
    'flask>=0.12.2',
    'GPyOpt>=1.2.5',
    'matplotlib'
]

if sys.version_info < (3, 4):
    REQUIRED += ['enum34']

# if args.parallel:
#     PARALLEL = ['pymongo>=3.5.1', 'drmaa>=0.7.7']
#     REQUIRED += PARALLEL
#     print("DRMAA path needs to be set e.g.:"
#           "export DRMAA_LIBRARY_PATH=/usr/lib/libdrmaa.so.1.0\n"
#           "See https://pypi.python.org/pypi/drmaa for details.")

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.rst' is present in your MANIFEST.in file!
with io.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = '\n' + f.read()


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPi via Twine…')
        os.system('twine upload dist/*')

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version='1.0.6',
    description=DESCRIPTION,
    long_description='',
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(),
    install_requires=REQUIRED,
    include_package_data=True,
    license='GPLv3',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    # $ setup.py publish support.
    keywords='hyperparameter optimization machine-learning',
    cmdclass={
        'upload': UploadCommand,
    },
)


