from __future__ import print_function
from __future__ import absolute_import
from sherpa.utils.testing_utils import load_dataset
from sherpa.utils.testing_utils import create_model_two as my_model
from sherpa.core import Hyperparameter
import sherpa.hparam_generators
import sherpa.algorithms
import sherpa.mainloop
import keras
import os
import pickle as pkl
from collections import defaultdict


def get_model(hparams, modelfile, historyfile):
    # Loads or creates a keras model
    if hparams is None or len(hparams) == 0:
        return load_keras_model(modelfile, historyfile)
    else:
        return create_keras_model(hparams)


def load_keras_model(modelfile, historyfile):
    # Restart from modelfile and historyfile.
    model = keras.models.load_model(modelfile)
    with open(historyfile, 'rb') as f:
        history = pkl.load(f)
    initial_epoch = len(history['loss'])
    return [model, history, initial_epoch]


def create_keras_model(hparams):
    model = my_model(hparams)
    history = defaultdict(list)
    initial_epoch = 0
    return [model, history, initial_epoch]


def save(model, modelfile, history, historyfile):
    # Save model and history files.
    model.save(modelfile)
    with open(historyfile, 'wb') as fid:
        pkl.dump(history, fid)


def update_history(partial_history, history):
    partial_history = partial_history.history
    for k in partial_history:
        history[k].extend(partial_history[k])
    assert 'loss' in history, 'Sherpa requires a loss to be defined in history.'


def main(modelfile, historyfile, hparams={}, epochs=1, verbose=2):
    model, history, initial_epoch = get_model(hparams, modelfile, historyfile)

    train_data, valid_data = load_dataset()

    partial_history = model.fit(x=train_data[0], y=train_data[1], batch_size=128,
                         validation_data=valid_data,
                         epochs=epochs + initial_epoch,
                         initial_epoch=initial_epoch,
                         verbose=verbose)

    # Update history
    update_history(partial_history, history)

    # Save model and history files.
    save(model, modelfile, history, historyfile)

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


    hp_generator = sherpa.hparam_generators.LatinHypercube

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