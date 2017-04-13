import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.initializers import glorot_uniform, Zeros
from subprocess import check_output
import re

def create_model(hparams):
    """
    Returns:
        A keras model for tests
    """
    # set seed for recreatability
    model = Sequential()
    model.add(Dense(hparams['num_units'], activation='relu', input_shape=(784,),
                      kernel_initializer=glorot_uniform(seed=1234), bias_initializer=Zeros()))
    model.add(Dense(10, activation='softmax',
                      kernel_initializer=glorot_uniform(seed=1234), bias_initializer=Zeros()))
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=hparams['lr']),
                  metrics=['accuracy'])
    return model


def create_model_two(hparams):
    """
    Returns:
        A keras model for tests
    """
    # set seed for recreatability
    model = Sequential()
    model.add(Dropout(rate=hparams['dropout'], input_shape=(784,)))
    model.add(Dense(100, activation=hparams['activation']))
    model.add(Dropout(rate=hparams['dropout']))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=hparams['learning_rate']),
                  metrics=['accuracy'])
    return model


def load_dataset(short=False):
    """
    Returns:
        A ready-to-train-on dataset for testing. Here, MNIST.
    """
    num_classes = 10

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)
    if short:
        return (x_train[:1000], y_train[:1000]), (x_test[:1000], y_test[:1000])
    else:
        return (x_train, y_train), (x_test, y_test)




def read_nvidia_smi(gpus=list(range(4)), cutoff=60):
    """
    Args:
        gpus: GPUs to be checked
        cutoff: Usage (in MiB) above which a gpu is declared to be in use

    Returns:
        True/False depending on whether those gpus are in use
    """

    s = check_output(["nvidia-smi"])
    matches = re.findall(r'([0-9]+)MiB / [0-9]+MiB', s)
    usage = map(int, matches)
    return all(gpu_usage >= cutoff for gpu_id, gpu_usage in enumerate(usage) if gpu_id in gpus)


def gpu_exists():
    """
    Returns:
        True/False depending on whether there is at least one GPU installed
    """
    try:
        check_output(["nvidia-smi"])
        return True
    except(OSError):
        return False