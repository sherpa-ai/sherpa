from __future__ import print_function
import os
import datetime
import argparse
import sherpa
from sherpa.hyperparameters import DistributionHyperparameter as Hyperparameter
from sherpa.scheduler import LocalScheduler,SGEScheduler

# Don't use gpu if we are just starting Sherpa.
if os.environ.get('KERAS_BACKEND') == 'theano':
    os.environ['THEANO_FLAGS'] = "floatX=float32,device=cpu,base_compiledir=~/.theano/cpu"
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ""


parser = argparse.ArgumentParser()
parser.add_argument('--sge', help='Use SGE', action='store_true')
FLAGS = parser.parse_args()


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
    alg = sherpa.algorithms.RandomSearch(samples=50, epochs=10, hp_ranges=hp_space)
    datetime_now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir = './output_{}'.format(datetime_now)  # All files written to here.

    if FLAGS.sge:
        env = '/home/lhertel/profiles/main.profile'  # Script specifying environment variables.
        opt = '-N myexample -P arcus.p -q arcus-ubuntu.q -l hostname=\'(arcus-7)\''  # SGE options.
        sched = SGEScheduler(dir=dir, environment=env, submit_options=opt)
    else:
        sched = LocalScheduler()  # Run on local machine without SGE.


    rval = sherpa.optimize(filename='mnist_convnet.py', algorithm=alg, dir=dir, overwrite=True, scheduler=sched, max_concurrent=4)
    print()
    print('Best results:')
    print(rval)


if __name__ == '__main__':
    run_sherpa()

