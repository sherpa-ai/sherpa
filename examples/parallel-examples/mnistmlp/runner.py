import os
import argparse
import sherpa
import datetime
from sherpa.schedulers import LocalScheduler,SGEScheduler
import sherpa.algorithms.bayesian_optimization as bayesian_optimization


def run_example(FLAGS):
    """
    Run parallel Sherpa optimization over a set of discrete hp combinations.
    """
    parameters = [sherpa.Continuous('lrinit', [0.001, 0.1], 'log'),
                  sherpa.Continuous('momentum', [0., 0.99]),
                  sherpa.Continuous('lrdecay', [1e-7, 1e-2], 'log'),
                  sherpa.Continuous('dropout', [0., 0.5])]
    
    if FLAGS.algorithm == 'BayesianOptimization':  
        print('Running GPyOpt')
        alg = bayesian_optimization.GPyOpt(max_concurrent=FLAGS.max_concurrent,
                                           model_type='GP_MCMC',
                                           acquisition_type='EI_MCMC',
                                           max_num_trials=150)
    elif FLAGS.algorithm == 'LocalSearch':
        print('Running Local Search')
        alg = sherpa.algorithms.LocalSearch(seed_configuration={'lrinit': 0.038,
                                                                'momentum': 0.92,
                                                                'lrdecay': 0.0001,
                                                                'dropout': 0.},
                                            perturbation_factors=(0.9, 1.1))
    else:
        print('Running Random Search')
        alg = sherpa.algorithms.RandomSearch(max_num_trials=150)

    if FLAGS.sge:
        assert FLAGS.env, "For SGE use, you need to set an environment path."
        # Submit to SGE queue.
        env = FLAGS.env  # Script specifying environment variables.
        opt = '-N MNISTExample -P {} -q {} -l {}'.format(FLAGS.P, FLAGS.q, FLAGS.l)
        sched = SGEScheduler(environment=env, submit_options=opt)
    else:
        # Run on local machine.
        sched = LocalScheduler()

    rval = sherpa.optimize(parameters=parameters,
                           algorithm=alg,
                           lower_is_better=True,
                           filename='trial.py',
                           output_dir='output_{}'.format(FLAGS.studyname),
                           scheduler=sched,
                           max_concurrent=FLAGS.max_concurrent)
    print('Best results:')
    print(rval)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sge', help='Run on SGE', action='store_true', default=False)
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
                        default='', type=str)
    parser.add_argument('--studyname', help='name for output folder', type=str, default='')
    parser.add_argument('--algorithm', type=str, default='BayesianOptimization')
    FLAGS = parser.parse_args()
    run_example(FLAGS)  # Sherpa optimization.
