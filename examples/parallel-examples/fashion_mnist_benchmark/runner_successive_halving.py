import os
import argparse
import sherpa
import datetime
import time
from sherpa.schedulers import LocalScheduler,SGEScheduler


def run_example(FLAGS):
    """
    Run parallel Sherpa optimization over a set of discrete hp combinations.
    """
    
    parameters = [sherpa.Continuous('learning_rate', [1e-5, 5e-1], 'log'),
                  sherpa.Continuous('decay', [1e-8, 1e-2], 'log'),
                  sherpa.Continuous('momentum', [0., 0.99]),
                  sherpa.Continuous('dropout', [0.0001, 0.7]),
                  sherpa.Ordinal('batch_size', [32, 64, 128, 256])]

    # 27 epochs to produce one finished 13 epoch model
    # => 54 epochs to produce one finished 26 epoch model (resource unit = 2epochs)
    # RS uses 100 models x 26 epochs = 2600 epochs.models
    # ASHA can produce 2600epochs.models//54epochs = 48 models
    algorithm = sherpa.algorithms.SuccessiveHalving(r=1, R=9, eta=3, s=0, max_finished_configs=48)

    # Run on local machine.
    scheduler = LocalScheduler()

    rval = sherpa.optimize(parameters=parameters,
                           algorithm=algorithm,
                           dashboard_port=FLAGS.port,
                           lower_is_better=False,
                           command='python fashion_mlp.py',
                           scheduler=scheduler,
                           verbose=0,
                           max_concurrent=FLAGS.concurrent,
                           output_dir='./output_successive_halving_{}'.format(
                               time.strftime("%Y-%m-%d--%H-%M-%S")))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--concurrent', type=int)
    parser.add_argument('--port', help='Dashboard port', type=int, default=8585)
    FLAGS = parser.parse_args()
    t0 = time.time()
    run_example(FLAGS)
    print("Time taken: ", time.time()-t0, "seconds")
