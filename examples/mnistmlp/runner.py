import os
import argparse
import sherpa
import datetime
from sherpa.schedulers import LocalScheduler,SGEScheduler


def run_example(FLAGS):
    """
    Run parallel Sherpa optimization over a set of discrete hp combinations.
    """
    parameters = [sherpa.Continuous('lrinit', [0.1, 0.01], 'log'),
                  sherpa.Continuous('momentum', [0., 0.99]),
                  sherpa.Continuous('lrdecay', [1e-2, 1e-7], 'log')]
    
    if FLAGS.algorithm == 'BayesianOptimization':  
        print('Running Bayesian Optimization')
        alg = sherpa.algorithms.BayesianOptimization(num_random_seeds=10,
                                                     max_num_trials=150)
    else:
        print('Running Random Search')
        alg = sherpa.algorithms.RandomSearch(max_num_trials=150)

    stopping_rule = None
    f = './trial.py' # Python script to run.
    dir = './output_' + FLAGS.studyname + '_' + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    if not FLAGS.local:
        # Submit to SGE queue.
        env = FLAGS.env  # Script specifying environment variables.
        opt = '-N MNISTExample -P {} -q {} -l {}'.format(FLAGS.P, FLAGS.q, FLAGS.l)
        sched = SGEScheduler(environment=env, submit_options=opt, output_dir=dir)
    else:
        # Run on local machine.
        sched = LocalScheduler()

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
    parser.add_argument('--local', help='Run locally', action='store_true', default=True)
    parser.add_argument('--max_concurrent',
                        help='Number of concurrent processes',
                        type=int, default=1)
    parser.add_argument('-P',
                        help="Specifies the project to which this  job  is  assigned.",
                        default='arcus_cpu.p')
    parser.add_argument('-q',
                        help='Defines a list of cluster queues or queue instances which may be used to execute this job.',
                        default='arcus.q')
    parser.add_argument('-l', help='the given resource list.',
                        default="hostname=\'(arcus-1|arcus-2|arcus-3|arcus-4|arcus-5|arcus-6|arcus-7|arcus-8|arcus-9)\'")
    parser.add_argument('--env', help='Your environment path.',
                        default='/home/lhertel/profiles/python3env.profile', type=str)
    parser.add_argument('--studyname', help='name for output folder', type=str, default='')
    parser.add_argument('--algorithm', type=str, default='BayesianOptimization')
    FLAGS = parser.parse_args()
    run_example(FLAGS)  # Sherpa optimization.
