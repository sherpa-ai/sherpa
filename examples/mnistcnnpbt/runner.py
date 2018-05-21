import sherpa
import sherpa.schedulers
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--local', help='Run locally', action='store_true',
                    default=True)
parser.add_argument('--max_concurrent',
                    help='Number of concurrent processes',
                    type=int, default=1)
parser.add_argument('-P',
                    help="Specifies the project to which this  job  is  assigned.",
                    default='arcus_gpu.p')
parser.add_argument('-q',
                    help='Defines a list of cluster queues or queue instances which may be used to execute this job.',
                    default='arcus.q')
parser.add_argument('-l', help='the given resource list.',
                    default="hostname=\'(arcus-1|arcus-2|arcus-3|arcus-4|arcus-5|arcus-6|arcus-7|arcus-8|arcus-9)\'")
parser.add_argument('--env', help='Your environment path.',
                    default='/home/lhertel/profiles/python3env.profile',
                    type=str)
FLAGS = parser.parse_args()


# Define Hyperparameter ranges
parameters = [sherpa.Continuous(name='lr', range=[0.005, 0.1], scale='log'),
              sherpa.Continuous(name='dropout', range=[0., 0.4]),
              sherpa.Ordinal(name='batch_size', range=[16, 32, 64])]

if FLAGS.algorithm == 'PBT':
    algorithm = sherpa.algorithms.PopulationBasedTraining(population_size=50,
                                                          parameter_range={
                                                              'lr':[0.0000001, 1.],
                                                              'batch_size':[16, 32, 64, 128, 256, 512]})
    parameters.append(sherpa.Choice(name='epochs', range=[3]))
    stoppingrule = None
else:
    parameters.append(sherpa.Continuous(name='lr_decay', range=[1e-4, 1e-7], scale='log'))
    parameters.append(sherpa.Choice(name='epochs', range=[25]))
    algorithm = sherpa.algorithms.BayesianOptimization(num_grid_points=2)
    # stoppingrule = sherpa.algorithms.MedianStoppingRule(min_trials=10,
    #                                                     min_iterations=8)

# The scheduler
if not FLAGS.local:
    env = FLAGS.env
    opt = '-N MNISTPBT -P {} -q {} -l {} -l gpu=1'.format(FLAGS.P, FLAGS.q, FLAGS.l)
    scheduler = sherpa.schedulers.SGEScheduler(environment=env, submit_options=opt)
else:
    scheduler = sherpa.schedulers.LocalScheduler()

# Running it all
sherpa.optimize(algorithm=algorithm,
                scheduler=scheduler,
                parameters=parameters,
                lower_is_better=True,
                filename="mnist_cnn.py",
                max_concurrent=FLAGS.max_concurrent,
                output_dir='./output')