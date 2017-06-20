import keras
import pickle as pkl
from collections import defaultdict


def load_model(hparams, my_model, model_file, history_file):
    # Loads or creates a keras model
    if hparams is None or len(hparams) == 0:
        return load_keras_model(model_file, history_file)
    else:
        return create_keras_model(hparams, my_model)


def load_keras_model(model_file, history_file):
    # Restart from model_file and history_file.
    model = keras.models.load_model(model_file)
    with open(history_file, 'rb') as f:
        history = pkl.load(f)
    initial_epoch = len(history['loss'])
    return [model, history, initial_epoch]


def create_keras_model(hparams, my_model):
    model = my_model(hparams)
    history = defaultdict(list)
    initial_epoch = 0
    return [model, history, initial_epoch]


def save_model(model, model_file, history, history_file):
    # Save model and history files.
    model.save(model_file)
    with open(history_file, 'wb') as fid:
        pkl.dump(history, fid)


def update_history(partial_history, history):
    partial_history = partial_history.history
    for k in partial_history:
        history[k].extend(partial_history[k])
    assert 'loss' in history, 'Sherpa requires a loss to be defined in history.'