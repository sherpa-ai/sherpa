from __future__ import print_function
import datetime
import argparse
import sherpa
from sherpa.hyperparameters import DistributionHyperparameter as Hyperparameter
from sherpa.schedulers import LocalScheduler,SGEScheduler


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
    alg = sherpa.algorithms.RandomSearch(samples=15, hp_ranges=hp_space)
    datetime_now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir = './output_{}'.format(datetime_now)  # All files written to here.

    if FLAGS.sge:
        env = '/home/lhertel/profiles/main.profile'  # Script specifying environment variables.
        opt = '-N sherpaMNIST -P {} -q {} -l {}'.format(FLAGS.P, FLAGS.q,
                                                        FLAGS.l)
        sched = SGEScheduler(environment=env, submit_options=opt)
    else:
        sched = LocalScheduler()  # Run on local machine without SGE.

    rval = sherpa.optimize(filename='mnist_convnet.py',
                           algorithm=alg,
                           dir=dir,
                           loss='val_loss',
                           overwrite=True,
                           scheduler=sched,
                           max_concurrent=FLAGS.max_concurrent)
    print()
    print('Best results:')
    print(rval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sge', help='Use SGE', action='store_true')
    parser.add_argument('--max_concurrent',
                        help='Number of concurrent processes',
                        type=int, default=1)
    parser.add_argument('-P',
                        help="Specifies the project to which this  job  is  assigned.",
                        default='arcus.p')
    parser.add_argument('-q',
                        help='Defines a list of cluster queues or queue instances which may be used to execute this job.',
                        default='arcus-ubuntu.q')
    parser.add_argument('-l', help='the given resource list.',
                        default="hostname=\'(arcus-5|arcus-6|arcus-8|arcus-9)\'")
    FLAGS = parser.parse_args()
    run_sherpa()

