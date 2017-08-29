from __future__ import print_function
from __future__ import absolute_import
from sherpa.utils.testing_utils import load_dataset
from sherpa.utils.testing_utils import create_model_two as my_model
from sherpa.hyperparameter import Hyperparameter
import sherpa.samplers
import sherpa.algorithms
import sherpa.mainloop
import os
from sherpa.utils.loading_and_saving_utils import load_model, update_history, save_model


def main(model_file, history_file, hparams={}, epochs=1, verbose=2):
    model, history, initial_epoch = load_model(hparams, my_model, model_file, history_file)

    train_data, valid_data = load_dataset()

    partial_history = model.fit(x=train_data[0], y=train_data[1], batch_size=128,
                         validation_data=valid_data,
                         epochs=epochs + initial_epoch,
                         initial_epoch=initial_epoch,
                         verbose=verbose)

    # Update history
    update_history(partial_history, history)

    # Save model and history files.
    save_model(model, model_file, history, history_file)

    return



def mnist_demo():
    '''
    Run Sherpa hyperparameter optimization.
    User may want to run this as a separate file.
    '''

    fname = os.path.basename(__file__)  # 'nn.py'

    # Hyperparameter space.
    my_hparam_ranges = [Hyperparameter(name='learning_rate',
                                       distr_args=(0.0001, 0.1),
                                       distribution='log-uniform'),
                        Hyperparameter(name='activation',
                                       distr_args=[('sigmoid', 'tanh', 'relu')],
                                       distribution='choice'),
                        Hyperparameter(name='dropout',
                                       distr_args=(0., 0.8),
                                       distribution='uniform')]


    hp_generator = sherpa.samplers.LatinHypercube

    # Algorithm used for optimization.
    alg = sherpa.algorithms.Hyperhack(samples=4, epochs_per_stage=2,
                                              stages=4,
                                              survival=0.5,
                                              hp_generator=hp_generator,
                                              hp_ranges=my_hparam_ranges,
                                              max_concurrent=10)

    session_dir = './session_dir'
    loop = sherpa.mainloop.MainLoop(fname=fname,
                                    algorithm=alg,
                                    dir=session_dir)
    loop.run()


if __name__ == '__main__':
    mnist_demo()
