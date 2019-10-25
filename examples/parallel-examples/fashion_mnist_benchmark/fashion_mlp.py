'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sherpa
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD


client = sherpa.Client()
trial = client.get_trial()
output_dir = os.environ.get("SHERPA_OUTPUT_DIR", '/tmp/')


# Loading data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

# Set the number of epochs
max_epochs = 26
if 'generation' in trial.parameters:
    epochs = trial.parameters['generation']
    initial_epoch = trial.parameters['generation']-1
elif 'resource' in trial.parameters:
    resource_unit = max_epochs//13
    initial_epoch = {1: 0, 3: 1, 9: 4}[trial.parameters['resource']] * resource_unit
    epochs = trial.parameters['resource'] * resource_unit + initial_epoch
else:
    epochs = max_epochs
    initial_epoch = 0

# Load or create model
if trial.parameters.get('load_from', '') != '':
    load_path = os.path.join(output_dir, trial.parameters['load_from'] + ".hdf5")
    model = load_model(load_path)
else:
    model = Sequential([Flatten(input_shape=(28, 28)),
                        Dense(512, activation='relu'),
                        Dropout(trial.parameters['dropout']),
                        Dense(512, activation='relu'),
                        Dropout(trial.parameters['dropout']),
                        Dense(10, activation='softmax')])

    optimizer = SGD(lr=trial.parameters['learning_rate'], momentum=trial.parameters['momentum'], decay=trial.parameters.get('decay', 0.))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

# Train model
history = model.fit(x_train, y_train,
                    batch_size=int(trial.parameters['batch_size']),
                    epochs=epochs,
                    verbose=2,
                    callbacks=[client.keras_send_metrics(trial,
                                                         objective_name='val_acc',
                                                         context_names=['val_loss', 'loss', 'acc'])],
                    validation_data=(x_test, y_test),
                    initial_epoch=initial_epoch)

if 'save_to' in trial.parameters:
    save_path = os.path.join(output_dir, trial.parameters['save_to'] + ".hdf5")
    model.save(save_path)
