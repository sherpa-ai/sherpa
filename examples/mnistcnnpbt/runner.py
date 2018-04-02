import sherpa
import sherpa.schedulers

# Define Hyperparameter ranges
parameters = [sherpa.Continuous(name='lr', range=[0.0001, 0.1], scale='log'),
              sherpa.Continuous(name='momentum', range=[0.5, 0.9]),
              sherpa.Ordinal(name='batch_size', range=[16, 32, 64])]

# The search algorithm - Random Search
algorithm = sherpa.algorithms.PopulationBasedTraining(population_size=50, parameter_range={'lr':[0.0000001, 1.],
                                                                                           'batch_size':[16, 32, 64, 128]})

# The SGD scheduler
scheduler = sherpa.schedulers.SGEScheduler(submit_options="-q arcus.q -P arcus_gpu.p -N MNISTPBT -l hostname=\'(arcus-6)\' -l gpu=1",
                                           environment='/home/lhertel/profiles/python3env.profile',
                                           output_dir='./output')

# Running it all
sherpa.optimize(algorithm=algorithm,
                scheduler=scheduler,
                parameters=parameters,
                lower_is_better=True,
                filename="mnist_cnn.py",
                max_concurrent=4,
                output_dir='./output')
