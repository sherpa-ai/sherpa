from setuptools import setup
from setuptools import find_packages

setup(name='sherpa',
      version='1.0',
      description='Hyperparameter Optimization',
      author='Baldi-Group',
      author_email='lhertel@uci.edu',
      url='https://gitlab.ics.uci.edu/uci-igb/sherpa',
      download_url='https://gitlab.ics.uci.edu/uci-igb/sherpa',
      license='',
      install_requires=['pandas',
                        'scipy'
                        ],
      package_data={'sherpa': ['README.md']},
      packages=find_packages())