import os
import pickle
from keras.models import load_model


class Experiment(object):
    def __init__(self, path, name, model=None):
        self.path = path
        self.name = name
        self.history = {'best_loss': 10000.,
                        'epochs': 0} if model else self._load_history()  # Like Keras history, accumulates meta data
        self.model = model or self._load_model()

    def _get_path(self, appendix):
        """
        Returns: absolute path to file with appendix

        """
        return os.path.join(self.path, self.name) + appendix

    def _load_model(self):
        """
        Returns: keras model

        """
        try:
            return load_model(self._get_path('_model.h5'))
        except OSError:
            raise IOError("Model file is missing from this experiment.")

    def _load_history(self):
        """
        Returns: loaded history object

        """
        try:
            with open(self._get_path('_history.pkl'), 'rb') as f:
                hist = pickle.load(f)
            return hist
        except IOError:
            raise IOError("You either tried to load an experiment that doesn't exist or the history file is missing.")

    def _save(self):
        """
        Saves model and history to path
        """
        self.model.save(self._get_path('_model.h5'))
        with open(self._get_path('_history.pkl'), 'wb') as f:
            pickle.dump(self.history, f)

    def fit(self, loss='val_loss', **kwargs):
        """
        Fits model for one iteration
        Args:
            args: Arguments of keras.model.fit()

        Returns:
            best performance so far and epochs so far

        """
        epochs = kwargs.pop('epochs') + self.history['epochs']

        if 'generator' in kwargs.keys():
            keras_hist = self.model.fit_generator(initial_epoch=self.history['epochs'], epochs=epochs, max_q_size=20, **kwargs)
        else:
            keras_hist = self.model.fit(initial_epoch=self.history[
                'epochs'], epochs=epochs, verbose=2, **kwargs)
        
        # Save best loss.
        this_loss = min(keras_hist.history[loss])
        self.history['best_loss'] = min(this_loss, self.history['best_loss'])
        self.history['epochs'] += len(keras_hist.epoch)
        self._save()
        return self.history['best_loss'], epochs

    # def __del__(self):
    #     self.model = None
