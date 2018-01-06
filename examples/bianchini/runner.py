import os
import argparse
import sherpa
from sherpa.schedulers import LocalScheduler,SGEScheduler


def run_example(FLAGS):
    """
    Run parallel Sherpa optimization over a set of discrete hp combinations.
    """
    # Iterate algorithm accepts dictionary containing lists of possible values. 
    hp_space = {'act': ['tanh', 'relu'],
                'lrinit': [0.1, 0.01],
                'momentum': [0.0],
                'lrdecay': [0.0],
                'arch': [[20,5], [20, 10], [10,10,10]],
                'epochs': [10],
                }
    parameters = sherpa.Parameter.grid(hp_space)

    alg = sherpa.algorithms.GridSearch()
    f = './bianchini.py' # Python script to run.
    dir = './output'       # All files written to here.

    if FLAGS.sge:
        # Submit to SGE queue.
        env = '/home/pjsadows/profiles/auto.profile'  # Script specifying environment variables.
        opt = '-N example -P {} -q {} -l {}'.format(FLAGS.P, FLAGS.q, FLAGS.l)
        sched = SGEScheduler(environment=env, submit_options=opt)
    else:
        # Run on local machine.
        sched = LocalScheduler()  # Run on local machine without SGE.

    rval = sherpa.optimize(parameters=parameters,
                           algorithm=alg,
                           output_dir=dir,
                           lower_is_better=True,
                           filename=f,
                           scheduler=sched,
                           max_concurrent=FLAGS.max_concurrent,
                           db_port=27005,
                           dashboard_port=8777)
    print()
    print('Best results:')
    print(rval)


if __name__=='__main__':
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
                        default="hostname=\'(arcus-7)\'")
    FLAGS = parser.parse_args()
    run_example(FLAGS)  # Sherpa optimization.

