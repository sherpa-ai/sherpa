# First Steps
In this tutorial we will write the code to train a shallow neural network to predict handwritten digits from the MNIST dataset using Keras and optimize it's hyperparameters using Hobbit.

## The Hyperparameters
First we need to define distributions and ranges for the hyperparameters we want to include in our model training. To do this we define a list of Hyperparameter objects, each one of these will be tuned afterwards. For each Hyperparameter we must assign a `name`, a tuple with the range of values to explore (`args`) and a `distribution` which indicates how to sample values within the defined range.

In this example we want to optimize three hyperparameters: the dropout rate, the activation function and the learning rate. We use uniform(0, 1) for the dropout rate, a discrete distribution over ```('sigmoid', 'tanh', 'relu')``` for the activation and a log-uniform for the learning rate. The log-uniform distribution samples uniformly on the log scale and then transforms back. You can see more information about the available distributions [here](https://github.com/LarsHH/hobbit/blob/master/hobbit/hparam_generators.py).


```python
from hobbit import Hyperparameter
my_hparam_ranges = [Hyperparameter(name='learning_rate', distribution='log-uniform', distr_args=(0.0001, 0.1)),
                    Hyperparameter(name='activation', distribution='choice', distr_args=[('sigmoid', 'tanh', 'relu')]),
                    Hyperparameter(name='dropout', distribution='uniform', distr_args=(0., 1.))]
```

## The model
With the principle of keeping Hobbit flexible we completely define the model we want to train inside a function which we call `my_model()`. This function takes a **dictionary of hyperparameters** `hparams` as its only argument and returns a **compiled Keras model**. Every model we train will follow this template and they will differ only in the values of the hyperparameters and how long they were trained.

Every optimizable hyperparameter within `my_model()` corresponds to one of the values defined before as a Hyperparameter object. Here we will specify where each hyperparameter goes and we will access it from the `hparams` dictionary like this 
`hparams['name_of_hyperparam']`


```python
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

def my_model(hparams):
    model = Sequential()
    model.add(Dropout(rate=hparams['dropout'], input_shape=(784,)))
    model.add(Dense(100, activation=hparams['activation']))
    model.add(Dropout(rate=hparams['dropout']))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=hparams['learning_rate']),
                  metrics=['accuracy'])
    return model
```


# The dataset
The dataset is loaded as a tuple of training and validation objects. These objects are usually numpy arrays with each sample as a row and each column as a feature. If there is some preprocessing for the data it has to be done before passing it to Hobbit. Here we load the MNIST dataset using Keras and do some simple preprocessing.


```python
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

my_dataset = (x_train, y_train), (x_test, y_test)
```

## Hyperband
We have all the ingredients. Now we just need to create the algorithm. As an argument you need to specify a repository directory. This is where Hobbit will store all of your models and the table with the results.


```python
from hobbit.algorithms import Hyperband
hband = Hyperband(model_function=my_model,
                  dataset=my_dataset,
                  hparam_ranges=my_hparam_ranges,
                  repo_dir='./my_test_repo')
```

## Running
Finally, let's run the complete pipeline. Hyperband has two parameters:
* **R**: The budget of epochs per stage
* **eta**: The cut-factor after each stage and which is also the factor by which training gets longer at every stage. For eta the theory-default is 3

To give you a feel for these here is an example for R=20 and eta=3. Here n_i is the number of configurations and r_i the number of epochs they are trained for. Hyperband makes multiple runs in which it does successive halving. smax corresponds to the current run.

    R = 20
    eta = 3

    smax=2
    n_0=9	r_0=2.22=2
    n_1=3	r_1=6.66=7
    n_2=1	r_2=20

    smax=1
    n_0=5	r_0=6.66=7
    n_1=1	r_1=20

    smax=0
    n_0=3	r_0=20

Now let's run our own Hyperband. If you're on a CPU this may take a few minutes.


```python
tab = hband.run(R=20, eta=3)
```

## Results
To access the results for all configurations and their respective test error you can either use the returned Pandas dataframe from the `run()` function or look at the CSV in your repository directory. Val Loss indicates the lowest seen validation loss for the configuration. The **Run** corresponds to the specific run it was trained at according to the Hyperband algorithm with run zero being the last and longest one. The **ID** is an identifier of the model within the run.


```python
tab
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hparams</th>
      <th>ID</th>
      <th>Run</th>
      <th>Val Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2-0</th>
      <td>{'learning_rate': 0.0013383297653888713, 'acti...</td>
      <td>0</td>
      <td>2</td>
      <td>0.079573</td>
    </tr>
    <tr>
      <th>2-1</th>
      <td>{'learning_rate': 0.00012199066883461425, 'act...</td>
      <td>1</td>
      <td>2</td>
      <td>0.504462</td>
    </tr>
    <tr>
      <th>2-2</th>
      <td>{'learning_rate': 0.00012022855306991257, 'act...</td>
      <td>2</td>
      <td>2</td>
      <td>0.361248</td>
    </tr>
    <tr>
      <th>2-3</th>
      <td>{'learning_rate': 0.01711893606634988, 'activa...</td>
      <td>3</td>
      <td>2</td>
      <td>0.110820</td>
    </tr>
    <tr>
      <th>2-4</th>
      <td>{'learning_rate': 0.0002600277942901727, 'acti...</td>
      <td>4</td>
      <td>2</td>
      <td>0.442517</td>
    </tr>
    <tr>
      <th>2-5</th>
      <td>{'learning_rate': 0.0004934370333196648, 'acti...</td>
      <td>5</td>
      <td>2</td>
      <td>0.287478</td>
    </tr>
    <tr>
      <th>2-6</th>
      <td>{'learning_rate': 0.007947367908290915, 'activ...</td>
      <td>6</td>
      <td>2</td>
      <td>1.002379</td>
    </tr>
    <tr>
      <th>2-7</th>
      <td>{'learning_rate': 0.000491890405781265, 'activ...</td>
      <td>7</td>
      <td>2</td>
      <td>0.280827</td>
    </tr>
    <tr>
      <th>2-8</th>
      <td>{'learning_rate': 0.0004272030665166832, 'acti...</td>
      <td>8</td>
      <td>2</td>
      <td>0.124675</td>
    </tr>
    <tr>
      <th>1-0</th>
      <td>{'learning_rate': 0.07198931288293847, 'activa...</td>
      <td>0</td>
      <td>1</td>
      <td>0.219331</td>
    </tr>
    <tr>
      <th>1-1</th>
      <td>{'learning_rate': 0.003255907153272381, 'activ...</td>
      <td>1</td>
      <td>1</td>
      <td>0.211909</td>
    </tr>
    <tr>
      <th>1-2</th>
      <td>{'learning_rate': 0.007926538077878733, 'activ...</td>
      <td>2</td>
      <td>1</td>
      <td>1.420097</td>
    </tr>
    <tr>
      <th>1-3</th>
      <td>{'learning_rate': 0.000888230245206736, 'activ...</td>
      <td>3</td>
      <td>1</td>
      <td>0.126166</td>
    </tr>
    <tr>
      <th>1-4</th>
      <td>{'learning_rate': 0.0011125712920494182, 'acti...</td>
      <td>4</td>
      <td>1</td>
      <td>0.077499</td>
    </tr>
    <tr>
      <th>0-0</th>
      <td>{'learning_rate': 0.018609130883389842, 'activ...</td>
      <td>0</td>
      <td>0</td>
      <td>0.148289</td>
    </tr>
    <tr>
      <th>0-1</th>
      <td>{'learning_rate': 0.003873098175478602, 'activ...</td>
      <td>1</td>
      <td>0</td>
      <td>0.098997</td>
    </tr>
    <tr>
      <th>0-2</th>
      <td>{'learning_rate': 0.0028061040541777402, 'acti...</td>
      <td>2</td>
      <td>0</td>
      <td>0.410078</td>
    </tr>
  </tbody>
</table>
</div>


## Using a Generator
Let's say you want to stream data using a generator. In that case you need to pass the function that returns a generator (not the generator object itself) to `generator_function`. You can pass a tuple if you need two different generator functions for training and testing. If they are just different by their arguments you can pass the relevant arguments via `train_gen_args` and `valid_gen_args` which accept either a dictionary or list/tuple. Hobbit also needs the number of steps / batches in training / testing. You pass these via `steps_per_epoch` and `validation_steps` similarly as in Keras.
```python
def example_generator(x, y, batch_size=100):
    num_samples = y.shape[0]
    num_batches = np.ceil(num_samples/batch_size).astype('int')
    while True:
        for i in range(num_batches):
            from_ = i*batch_size
            to_ = min((i+1)*batch_size, num_samples)
            yield x[from_:to_], y[from_:to_]
```

Then set up Hyperband:

```python
hband = Hyperband(model_function=my_model,
                  hparam_ranges=my_hparam_ranges,
                  repo_dir='./my_test_repo',
                  generator_function=example_generator,
                  train_gen_args=(x_train, y_train, 100),
                  valid_gen_args=(x_test, y_test, 100),
                  steps_per_epoch=x_train.shape[0]//100,
                  validation_steps=x_test.shape[0]//100)
```