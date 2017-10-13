from __future__ import print_function
import os
import sherpa
from sherpa.hyperparameters import DistributionHyperparameter as Hyperparameter
from sherpa.scheduler import LocalScheduler,SGEScheduler

# Don't use gpu if we are just starting Sherpa.
if os.environ['KERAS_BACKEND'] == 'theano':
    os.environ['THEANO_FLAGS'] = "floatX=float32,device=cpu,base_compiledir=~/.theano/cpu"
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ""


def run_sherpa():
    '''
    Run Sherpa hyperparameter optimization.
    User may want to run this as a separate file.
    '''
    # Hyperparameter space.
    hp_space = [
        Hyperparameter(name='num_filters', distribution='choice', dist_args=[32, 64, 96]),
        Hyperparameter(name='filter_size', distribution='choice', dist_args=[3, 5, 7]),
        Hyperparameter(name='dropout', distribution='uniform', dist_args=[0.0001, 0.6]),
        Hyperparameter(name='activation', distribution='choice', dist_args=['tanh', 'relu', 'sigmoid']),
    ]

    # Algorithm used for optimization.
    alg = sherpa.algorithms.RandomSearch(samples=10, epochs=10, hp_ranges=hp_space)
    # alg  = sherpa.algorithms.RandomSearch(samples=100, epochs=1, hp_ranges=hp_ranges, max_concurrent=10)

    dir = './output'  # All files written to here.
    sched = LocalScheduler()  # Run on local machine without SGE.
    rval = sherpa.optimize(filename='mnist_convnet.py', algorithm=alg, dir=dir, overwrite=True, scheduler=sched, max_concurrent=3)
    print()
    print('Best results:')
    print(rval)


if __name__ == '__main__':
    run_sherpa()  # Sherpa optimization.

