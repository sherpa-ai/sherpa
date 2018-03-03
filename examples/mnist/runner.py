import os
import argparse
import sherpa
from sherpa.schedulers import LocalScheduler,SGEScheduler


def run_example(FLAGS):
    """
    Run parallel Sherpa optimization over a set of discrete hp combinations.
    """
    # Iterate algorithm accepts dictionary containing lists of possible values.
    # hp_space = {'act': ['tanh', 'relu'],
    #             'lrinit': [0.1, 0.01],
    #             'momentum': [0.0],
    #             'lrdecay': [0.0],
    #             'arch': [[20,5], [20, 10], [10,10,10]],
    #             'epochs': [20],
    #             }
    # parameters = sherpa.Parameter.grid(hp_space)
    parameters = [sherpa.Continuous('lrinit', [0.1, 0.01], 'log'),
                  sherpa.Continuous('momentum', [0., 0.99]),
                  sherpa.Continuous('lrdecay', [1e-2, 1e-7], 'log'),
                  sherpa.Choice('act', ['tanh']),
                  sherpa.Choice('arch', [[100, 100]]),
                  sherpa.Choice('epochs', [10])]


    # alg = sherpa.algorithms.GridSearch()
    alg = sherpa.algorithms.GaussianProcessEI(num_random_seeds=10,
                                              max_num_trials=150)
    # stopping_rule = sherpa.algorithms.MedianStoppingRule(min_iterations=10, min_trials=5)
    stopping_rule = None
    f = './trial.py' # Python script to run.
    dir = './output'       # All files written to here.

    if not FLAGS.local:
        # Submit to SGE queue.
        # env = '/home/pjsadows/profiles/auto.profile'  # Script specifying environment variables.
        env = FLAGS.env
        opt = '-N example -P {} -q {} -l {}'.format(FLAGS.P, FLAGS.q, FLAGS.l)
        sched = SGEScheduler(environment=env, submit_options=opt, output_dir=dir)
    else:
        # Run on local machine.
        sched = LocalScheduler()  # Run on local machine without SGE.

    rval = sherpa.optimize(parameters=parameters,
                           algorithm=alg,
                           stopping_rule=stopping_rule,
                           output_dir=dir,
                           lower_is_better=False,
                           filename=f,
                           scheduler=sched,
                           max_concurrent=FLAGS.max_concurrent)
    print()
    print('Best results:')
    print(rval)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', help='Run locally', action='store_true', default=False)
    parser.add_argument('--max_concurrent',
                        help='Number of concurrent processes',
                        type=int, default=1)
    parser.add_argument('--P',
                        help="Specifies the project to which this  job  is  assigned.",
                        default='arcus_cpu.p')
    parser.add_argument('--q',
                        help='Defines a list of cluster queues or queue instances which may be used to execute this job.',
                        default='arcus.q')
    parser.add_argument('--l', help='the given resource list.',
                        default="hostname=\'(arcus-2)\'")
    parser.add_argument('--env', help='Your environment path.',
                        default='/home/lhertel/profiles/python3env.profile', type=str)
    FLAGS = parser.parse_args()
    run_example(FLAGS)  # Sherpa optimization.
