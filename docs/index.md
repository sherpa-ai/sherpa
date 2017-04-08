# Hobbit
Welcome to Hobbit aka. Hyperparameter Optimization By BandITs. Hobbit's goal is to give you a tool for fast and easy
hyperparameter search. This notebook is going to show you how to run a hyperparameter optimization on a simple model
and dataset.

## Motivation
The main idea behind Hobbit is to make an extremely easy to use, Keras compatible, framework for hyperparameter
optimization. It works by specifying a model template with modifyable hyperparameters, then it automatically trains as
many models as wanted and as long as needed using ranges specified by the user for the tuneable hyperparameters.