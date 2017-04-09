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

    Using TensorFlow backend.


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

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/2
    60000/60000 [==============================] - 3s - loss: 0.4150 - acc: 0.8760 - val_loss: 0.1885 - val_acc: 0.9462
    Epoch 2/2
    60000/60000 [==============================] - 2s - loss: 0.2318 - acc: 0.9318 - val_loss: 0.1363 - val_acc: 0.9600
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/2
    60000/60000 [==============================] - 3s - loss: 1.7855 - acc: 0.4016 - val_loss: 0.6827 - val_acc: 0.8353
    Epoch 2/2
    60000/60000 [==============================] - 3s - loss: 1.1040 - acc: 0.6311 - val_loss: 0.5045 - val_acc: 0.8602
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/2
    60000/60000 [==============================] - 4s - loss: 1.2102 - acc: 0.6269 - val_loss: 0.4898 - val_acc: 0.8783
    Epoch 2/2
    60000/60000 [==============================] - 3s - loss: 0.6096 - acc: 0.8167 - val_loss: 0.3612 - val_acc: 0.9060
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/2
    60000/60000 [==============================] - 3s - loss: 0.3670 - acc: 0.8883 - val_loss: 0.1583 - val_acc: 0.9493
    Epoch 2/2
    60000/60000 [==============================] - 6s - loss: 0.2433 - acc: 0.9277 - val_loss: 0.1364 - val_acc: 0.9587
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/2
    60000/60000 [==============================] - 1796s - loss: 1.6579 - acc: 0.4532 - val_loss: 0.5450 - val_acc: 0.8569
    Epoch 2/2
    60000/60000 [==============================] - 2s - loss: 1.0364 - acc: 0.6585 - val_loss: 0.4425 - val_acc: 0.8751
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/2
    60000/60000 [==============================] - 3s - loss: 0.7051 - acc: 0.7810 - val_loss: 0.3255 - val_acc: 0.9070
    Epoch 2/2
    60000/60000 [==============================] - 2s - loss: 0.4682 - acc: 0.8560 - val_loss: 0.2875 - val_acc: 0.9174
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/2
    60000/60000 [==============================] - 3s - loss: 1.9400 - acc: 0.3203 - val_loss: 1.1151 - val_acc: 0.7372
    Epoch 2/2
    60000/60000 [==============================] - 2s - loss: 1.8043 - acc: 0.3731 - val_loss: 1.0024 - val_acc: 0.7524
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/2
    60000/60000 [==============================] - 3s - loss: 0.6797 - acc: 0.7923 - val_loss: 0.3231 - val_acc: 0.9080
    Epoch 2/2
    60000/60000 [==============================] - 2s - loss: 0.4484 - acc: 0.8637 - val_loss: 0.2808 - val_acc: 0.9188
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/2
    60000/60000 [==============================] - 3s - loss: 0.7018 - acc: 0.7843 - val_loss: 0.2968 - val_acc: 0.9162
    Epoch 2/2
    60000/60000 [==============================] - 3s - loss: 0.3897 - acc: 0.8818 - val_loss: 0.2301 - val_acc: 0.9330
    Train on 60000 samples, validate on 10000 samples
    Epoch 3/9
    60000/60000 [==============================] - 3s - loss: 0.1887 - acc: 0.9446 - val_loss: 0.1150 - val_acc: 0.9666
    Epoch 4/9
    60000/60000 [==============================] - 2s - loss: 0.1682 - acc: 0.9494 - val_loss: 0.1040 - val_acc: 0.9693
    Epoch 5/9
    60000/60000 [==============================] - 3s - loss: 0.1530 - acc: 0.9544 - val_loss: 0.0986 - val_acc: 0.9705
    Epoch 6/9
    60000/60000 [==============================] - 2s - loss: 0.1430 - acc: 0.9573 - val_loss: 0.0936 - val_acc: 0.9740
    Epoch 7/9
    60000/60000 [==============================] - 2s - loss: 0.1361 - acc: 0.9595 - val_loss: 0.0957 - val_acc: 0.9740
    Epoch 8/9
    60000/60000 [==============================] - 2s - loss: 0.1330 - acc: 0.9604 - val_loss: 0.0881 - val_acc: 0.9753
    Epoch 9/9
    60000/60000 [==============================] - 2s - loss: 0.1290 - acc: 0.9619 - val_loss: 0.0893 - val_acc: 0.9763
    Train on 60000 samples, validate on 10000 samples
    Epoch 3/9
    60000/60000 [==============================] - 3s - loss: 0.2233 - acc: 0.9360 - val_loss: 0.1253 - val_acc: 0.9666
    Epoch 4/9
    60000/60000 [==============================] - 2s - loss: 0.2138 - acc: 0.9392 - val_loss: 0.1252 - val_acc: 0.9636
    Epoch 5/9
    60000/60000 [==============================] - 2s - loss: 0.2089 - acc: 0.9420 - val_loss: 0.1328 - val_acc: 0.9633
    Epoch 6/9
    60000/60000 [==============================] - 2s - loss: 0.2046 - acc: 0.9441 - val_loss: 0.1164 - val_acc: 0.9677
    Epoch 7/9
    60000/60000 [==============================] - 2s - loss: 0.2018 - acc: 0.9448 - val_loss: 0.1111 - val_acc: 0.9704
    Epoch 8/9
    60000/60000 [==============================] - 2s - loss: 0.1993 - acc: 0.9458 - val_loss: 0.1108 - val_acc: 0.9713
    Epoch 9/9
    60000/60000 [==============================] - 2s - loss: 0.1964 - acc: 0.9473 - val_loss: 0.1123 - val_acc: 0.9714
    Train on 60000 samples, validate on 10000 samples
    Epoch 3/9
    60000/60000 [==============================] - 3s - loss: 0.3254 - acc: 0.9009 - val_loss: 0.1908 - val_acc: 0.9457
    Epoch 4/9
    60000/60000 [==============================] - 2s - loss: 0.2878 - acc: 0.9125 - val_loss: 0.1671 - val_acc: 0.9503
    Epoch 5/9
    60000/60000 [==============================] - 2s - loss: 0.2631 - acc: 0.9219 - val_loss: 0.1511 - val_acc: 0.9556
    Epoch 6/9
    60000/60000 [==============================] - 2s - loss: 0.2450 - acc: 0.9275 - val_loss: 0.1419 - val_acc: 0.9591
    Epoch 7/9
    60000/60000 [==============================] - 2s - loss: 0.2349 - acc: 0.9296 - val_loss: 0.1334 - val_acc: 0.9618
    Epoch 8/9
    60000/60000 [==============================] - 2s - loss: 0.2261 - acc: 0.9335 - val_loss: 0.1276 - val_acc: 0.9646
    Epoch 9/9
    60000/60000 [==============================] - 2s - loss: 0.2194 - acc: 0.9348 - val_loss: 0.1247 - val_acc: 0.9640
    Train on 60000 samples, validate on 10000 samples
    Epoch 10/29
    60000/60000 [==============================] - 3s - loss: 0.1240 - acc: 0.9636 - val_loss: 0.0939 - val_acc: 0.9751
    Epoch 11/29
    60000/60000 [==============================] - 3s - loss: 0.1244 - acc: 0.9639 - val_loss: 0.0822 - val_acc: 0.9774
    Epoch 12/29
    60000/60000 [==============================] - 3s - loss: 0.1194 - acc: 0.9650 - val_loss: 0.0888 - val_acc: 0.9769
    Epoch 13/29
    60000/60000 [==============================] - 3s - loss: 0.1187 - acc: 0.9650 - val_loss: 0.0835 - val_acc: 0.9785
    Epoch 14/29
    60000/60000 [==============================] - 3s - loss: 0.1187 - acc: 0.9657 - val_loss: 0.0867 - val_acc: 0.9779
    Epoch 15/29
    60000/60000 [==============================] - 3s - loss: 0.1132 - acc: 0.9669 - val_loss: 0.0807 - val_acc: 0.9783
    Epoch 16/29
    60000/60000 [==============================] - 4s - loss: 0.1137 - acc: 0.9669 - val_loss: 0.0802 - val_acc: 0.9790
    Epoch 17/29
    60000/60000 [==============================] - 3s - loss: 0.1139 - acc: 0.9669 - val_loss: 0.0858 - val_acc: 0.9783
    Epoch 18/29
    60000/60000 [==============================] - 3s - loss: 0.1114 - acc: 0.9676 - val_loss: 0.0847 - val_acc: 0.9790
    Epoch 19/29
    60000/60000 [==============================] - 3s - loss: 0.1109 - acc: 0.9681 - val_loss: 0.0863 - val_acc: 0.9780
    Epoch 20/29
    60000/60000 [==============================] - 3s - loss: 0.1128 - acc: 0.9678 - val_loss: 0.0796 - val_acc: 0.9800
    Epoch 21/29
    60000/60000 [==============================] - 3s - loss: 0.1115 - acc: 0.9687 - val_loss: 0.0898 - val_acc: 0.9783
    Epoch 22/29
    60000/60000 [==============================] - 3s - loss: 0.1102 - acc: 0.9690 - val_loss: 0.0848 - val_acc: 0.9785
    Epoch 23/29
    60000/60000 [==============================] - 3s - loss: 0.1100 - acc: 0.9690 - val_loss: 0.0857 - val_acc: 0.9791
    Epoch 24/29
    60000/60000 [==============================] - 3s - loss: 0.1136 - acc: 0.9681 - val_loss: 0.0846 - val_acc: 0.9785
    Epoch 25/29
    60000/60000 [==============================] - 3s - loss: 0.1098 - acc: 0.9695 - val_loss: 0.0901 - val_acc: 0.9788
    Epoch 26/29
    60000/60000 [==============================] - 3s - loss: 0.1113 - acc: 0.9687 - val_loss: 0.0909 - val_acc: 0.9794
    Epoch 27/29
    60000/60000 [==============================] - 3s - loss: 0.1075 - acc: 0.9701 - val_loss: 0.0864 - val_acc: 0.9796
    Epoch 28/29
    60000/60000 [==============================] - 3s - loss: 0.1082 - acc: 0.9703 - val_loss: 0.0859 - val_acc: 0.9795
    Epoch 29/29
    60000/60000 [==============================] - 3s - loss: 0.1069 - acc: 0.9711 - val_loss: 0.0897 - val_acc: 0.9803
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/7
    60000/60000 [==============================] - 3s - loss: 0.7848 - acc: 0.7782 - val_loss: 0.2634 - val_acc: 0.9253
    Epoch 2/7
    60000/60000 [==============================] - 2s - loss: 0.5881 - acc: 0.8467 - val_loss: 0.2748 - val_acc: 0.9263
    Epoch 3/7
    60000/60000 [==============================] - 2s - loss: 0.5668 - acc: 0.8597 - val_loss: 0.2366 - val_acc: 0.9393
    Epoch 4/7
    60000/60000 [==============================] - 2s - loss: 0.5528 - acc: 0.8651 - val_loss: 0.2325 - val_acc: 0.9385
    Epoch 5/7
    60000/60000 [==============================] - 2s - loss: 0.5413 - acc: 0.8711 - val_loss: 0.2193 - val_acc: 0.9395
    Epoch 6/7
    60000/60000 [==============================] - 2s - loss: 0.5352 - acc: 0.8734 - val_loss: 0.2305 - val_acc: 0.9445
    Epoch 7/7
    60000/60000 [==============================] - 2s - loss: 0.5239 - acc: 0.8770 - val_loss: 0.2384 - val_acc: 0.9377
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/7
    60000/60000 [==============================] - 3s - loss: 0.6633 - acc: 0.7884 - val_loss: 0.2842 - val_acc: 0.9173
    Epoch 2/7
    60000/60000 [==============================] - 2s - loss: 0.5274 - acc: 0.8347 - val_loss: 0.2584 - val_acc: 0.9240
    Epoch 3/7
    60000/60000 [==============================] - 2s - loss: 0.4970 - acc: 0.8466 - val_loss: 0.2392 - val_acc: 0.9283
    Epoch 4/7
    60000/60000 [==============================] - 2s - loss: 0.4754 - acc: 0.8521 - val_loss: 0.2274 - val_acc: 0.9310
    Epoch 5/7
    60000/60000 [==============================] - 2s - loss: 0.4691 - acc: 0.8548 - val_loss: 0.2158 - val_acc: 0.9368
    Epoch 6/7
    60000/60000 [==============================] - 2s - loss: 0.4569 - acc: 0.8597 - val_loss: 0.2134 - val_acc: 0.9358
    Epoch 7/7
    60000/60000 [==============================] - 2s - loss: 0.4457 - acc: 0.8634 - val_loss: 0.2119 - val_acc: 0.9361
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/7
    60000/60000 [==============================] - 3s - loss: 2.5374 - acc: 0.1547 - val_loss: 1.6141 - val_acc: 0.5675
    Epoch 2/7
    60000/60000 [==============================] - 2s - loss: 2.3243 - acc: 0.1889 - val_loss: 1.4994 - val_acc: 0.5680
    Epoch 3/7
    60000/60000 [==============================] - 2s - loss: 2.2891 - acc: 0.2054 - val_loss: 1.4632 - val_acc: 0.5597
    Epoch 4/7
    60000/60000 [==============================] - 2s - loss: 2.2771 - acc: 0.2129 - val_loss: 1.4780 - val_acc: 0.5704
    Epoch 5/7
    60000/60000 [==============================] - 2s - loss: 2.2665 - acc: 0.2131 - val_loss: 1.4547 - val_acc: 0.5981
    Epoch 6/7
    60000/60000 [==============================] - 2s - loss: 2.2559 - acc: 0.2193 - val_loss: 1.4463 - val_acc: 0.5676
    Epoch 7/7
    60000/60000 [==============================] - 2s - loss: 2.2445 - acc: 0.2245 - val_loss: 1.4201 - val_acc: 0.6015
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/7
    60000/60000 [==============================] - 3s - loss: 0.6224 - acc: 0.8081 - val_loss: 0.2482 - val_acc: 0.9316
    Epoch 2/7
    60000/60000 [==============================] - 2s - loss: 0.3783 - acc: 0.8852 - val_loss: 0.1976 - val_acc: 0.9444
    Epoch 3/7
    60000/60000 [==============================] - 2s - loss: 0.3274 - acc: 0.9012 - val_loss: 0.1644 - val_acc: 0.9530
    Epoch 4/7
    60000/60000 [==============================] - 2s - loss: 0.2971 - acc: 0.9095 - val_loss: 0.1508 - val_acc: 0.9564
    Epoch 5/7
    60000/60000 [==============================] - 2s - loss: 0.2797 - acc: 0.9157 - val_loss: 0.1375 - val_acc: 0.9611
    Epoch 6/7
    60000/60000 [==============================] - 2s - loss: 0.2710 - acc: 0.9193 - val_loss: 0.1294 - val_acc: 0.9627
    Epoch 7/7
    60000/60000 [==============================] - 2s - loss: 0.2614 - acc: 0.9227 - val_loss: 0.1262 - val_acc: 0.9633
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/7
    60000/60000 [==============================] - 3s - loss: 0.5043 - acc: 0.8756 - val_loss: 0.2684 - val_acc: 0.9230
    Epoch 2/7
    60000/60000 [==============================] - 2s - loss: 0.2454 - acc: 0.9302 - val_loss: 0.2100 - val_acc: 0.9375
    Epoch 3/7
    60000/60000 [==============================] - 2s - loss: 0.1968 - acc: 0.9432 - val_loss: 0.1753 - val_acc: 0.9492
    Epoch 4/7
    60000/60000 [==============================] - 2s - loss: 0.1658 - acc: 0.9518 - val_loss: 0.1542 - val_acc: 0.9543
    Epoch 5/7
    60000/60000 [==============================] - 2s - loss: 0.1424 - acc: 0.9586 - val_loss: 0.1353 - val_acc: 0.9589
    Epoch 6/7
    60000/60000 [==============================] - 2s - loss: 0.1258 - acc: 0.9635 - val_loss: 0.1268 - val_acc: 0.9628
    Epoch 7/7
    60000/60000 [==============================] - 2s - loss: 0.1124 - acc: 0.9673 - val_loss: 0.1175 - val_acc: 0.9657
    Train on 60000 samples, validate on 10000 samples
    Epoch 8/27
    60000/60000 [==============================] - 3s - loss: 0.1015 - acc: 0.9708 - val_loss: 0.1088 - val_acc: 0.9673
    Epoch 9/27
    60000/60000 [==============================] - 2s - loss: 0.0924 - acc: 0.9733 - val_loss: 0.1032 - val_acc: 0.9695
    Epoch 10/27
    60000/60000 [==============================] - 2s - loss: 0.0852 - acc: 0.9756 - val_loss: 0.0992 - val_acc: 0.9703
    Epoch 11/27
    60000/60000 [==============================] - 2s - loss: 0.0781 - acc: 0.9774 - val_loss: 0.0943 - val_acc: 0.9734
    Epoch 12/27
    60000/60000 [==============================] - 2s - loss: 0.0727 - acc: 0.9791 - val_loss: 0.0932 - val_acc: 0.9715
    Epoch 13/27
    60000/60000 [==============================] - 2s - loss: 0.0677 - acc: 0.9806 - val_loss: 0.0896 - val_acc: 0.9735
    Epoch 14/27
    60000/60000 [==============================] - 2s - loss: 0.0632 - acc: 0.9825 - val_loss: 0.0861 - val_acc: 0.9745
    Epoch 15/27
    60000/60000 [==============================] - 3s - loss: 0.0592 - acc: 0.9831 - val_loss: 0.0879 - val_acc: 0.9736
    Epoch 16/27
    60000/60000 [==============================] - 2s - loss: 0.0555 - acc: 0.9842 - val_loss: 0.0827 - val_acc: 0.9756
    Epoch 17/27
    60000/60000 [==============================] - 3s - loss: 0.0522 - acc: 0.9852 - val_loss: 0.0829 - val_acc: 0.9753
    Epoch 18/27
    60000/60000 [==============================] - 3s - loss: 0.0490 - acc: 0.9866 - val_loss: 0.0820 - val_acc: 0.9753
    Epoch 19/27
    60000/60000 [==============================] - 2s - loss: 0.0466 - acc: 0.9870 - val_loss: 0.0807 - val_acc: 0.9753
    Epoch 20/27
    60000/60000 [==============================] - 2s - loss: 0.0438 - acc: 0.9876 - val_loss: 0.0810 - val_acc: 0.9757
    Epoch 21/27
    60000/60000 [==============================] - 3s - loss: 0.0416 - acc: 0.9883 - val_loss: 0.0804 - val_acc: 0.9763
    Epoch 22/27
    60000/60000 [==============================] - 3s - loss: 0.0390 - acc: 0.9893 - val_loss: 0.0803 - val_acc: 0.9766
    Epoch 23/27
    60000/60000 [==============================] - 3s - loss: 0.0369 - acc: 0.9897 - val_loss: 0.0788 - val_acc: 0.9770
    Epoch 24/27
    60000/60000 [==============================] - 3s - loss: 0.0355 - acc: 0.9899 - val_loss: 0.0794 - val_acc: 0.9777
    Epoch 25/27
    60000/60000 [==============================] - 3s - loss: 0.0338 - acc: 0.9905 - val_loss: 0.0797 - val_acc: 0.9774
    Epoch 26/27
    60000/60000 [==============================] - 3s - loss: 0.0317 - acc: 0.9915 - val_loss: 0.0808 - val_acc: 0.9771
    Epoch 27/27
    60000/60000 [==============================] - 2s - loss: 0.0305 - acc: 0.9917 - val_loss: 0.0775 - val_acc: 0.9782
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/20
    60000/60000 [==============================] - 3s - loss: 0.4937 - acc: 0.8535 - val_loss: 0.2463 - val_acc: 0.9267
    Epoch 2/20
    60000/60000 [==============================] - 3s - loss: 0.3701 - acc: 0.8913 - val_loss: 0.1996 - val_acc: 0.9413
    Epoch 3/20
    60000/60000 [==============================] - 3s - loss: 0.3380 - acc: 0.9018 - val_loss: 0.1878 - val_acc: 0.9472
    Epoch 4/20
    60000/60000 [==============================] - 3s - loss: 0.3238 - acc: 0.9060 - val_loss: 0.1793 - val_acc: 0.9484
    Epoch 5/20
    60000/60000 [==============================] - 3s - loss: 0.3135 - acc: 0.9095 - val_loss: 0.1768 - val_acc: 0.9489
    Epoch 6/20
    60000/60000 [==============================] - 3s - loss: 0.3078 - acc: 0.9123 - val_loss: 0.1676 - val_acc: 0.9530
    Epoch 7/20
    60000/60000 [==============================] - 3s - loss: 0.3034 - acc: 0.9135 - val_loss: 0.1779 - val_acc: 0.9506
    Epoch 8/20
    60000/60000 [==============================] - 2s - loss: 0.2986 - acc: 0.9137 - val_loss: 0.1816 - val_acc: 0.9484
    Epoch 9/20
    60000/60000 [==============================] - 2s - loss: 0.2906 - acc: 0.9159 - val_loss: 0.1768 - val_acc: 0.9508
    Epoch 10/20
    60000/60000 [==============================] - 2s - loss: 0.2923 - acc: 0.9165 - val_loss: 0.1790 - val_acc: 0.9479
    Epoch 11/20
    60000/60000 [==============================] - 2s - loss: 0.2870 - acc: 0.9167 - val_loss: 0.1626 - val_acc: 0.9553
    Epoch 12/20
    60000/60000 [==============================] - 2s - loss: 0.2824 - acc: 0.9199 - val_loss: 0.1566 - val_acc: 0.9523
    Epoch 13/20
    60000/60000 [==============================] - 2s - loss: 0.2858 - acc: 0.9180 - val_loss: 0.1483 - val_acc: 0.9572
    Epoch 14/20
    60000/60000 [==============================] - 2s - loss: 0.2809 - acc: 0.9187 - val_loss: 0.1494 - val_acc: 0.9578
    Epoch 15/20
    60000/60000 [==============================] - 2s - loss: 0.2802 - acc: 0.9204 - val_loss: 0.1583 - val_acc: 0.9571
    Epoch 16/20
    60000/60000 [==============================] - 2s - loss: 0.2815 - acc: 0.9202 - val_loss: 0.1570 - val_acc: 0.9559
    Epoch 17/20
    60000/60000 [==============================] - 2s - loss: 0.2789 - acc: 0.9206 - val_loss: 0.1536 - val_acc: 0.9556
    Epoch 18/20
    60000/60000 [==============================] - 2s - loss: 0.2723 - acc: 0.9218 - val_loss: 0.1543 - val_acc: 0.9550
    Epoch 19/20
    60000/60000 [==============================] - 2s - loss: 0.2742 - acc: 0.9216 - val_loss: 0.1665 - val_acc: 0.9518
    Epoch 20/20
    60000/60000 [==============================] - 2s - loss: 0.2759 - acc: 0.9226 - val_loss: 0.1591 - val_acc: 0.9564
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/20
    60000/60000 [==============================] - 3s - loss: 0.3963 - acc: 0.8789 - val_loss: 0.1899 - val_acc: 0.9456
    Epoch 2/20
    60000/60000 [==============================] - 2s - loss: 0.2568 - acc: 0.9203 - val_loss: 0.1532 - val_acc: 0.9548
    Epoch 3/20
    60000/60000 [==============================] - 2s - loss: 0.2249 - acc: 0.9297 - val_loss: 0.1321 - val_acc: 0.9619
    Epoch 4/20
    60000/60000 [==============================] - 2s - loss: 0.2120 - acc: 0.9354 - val_loss: 0.1210 - val_acc: 0.9648
    Epoch 5/20
    60000/60000 [==============================] - 2s - loss: 0.1999 - acc: 0.9388 - val_loss: 0.1211 - val_acc: 0.9633
    Epoch 6/20
    60000/60000 [==============================] - 2s - loss: 0.1910 - acc: 0.9413 - val_loss: 0.1097 - val_acc: 0.9682
    Epoch 7/20
    60000/60000 [==============================] - 2s - loss: 0.1880 - acc: 0.9434 - val_loss: 0.1083 - val_acc: 0.9689
    Epoch 8/20
    60000/60000 [==============================] - 2s - loss: 0.1856 - acc: 0.9436 - val_loss: 0.1026 - val_acc: 0.9695
    Epoch 9/20
    60000/60000 [==============================] - 2s - loss: 0.1790 - acc: 0.9451 - val_loss: 0.1093 - val_acc: 0.9678
    Epoch 10/20
    60000/60000 [==============================] - 2s - loss: 0.1761 - acc: 0.9462 - val_loss: 0.1080 - val_acc: 0.9691
    Epoch 11/20
    60000/60000 [==============================] - 2s - loss: 0.1760 - acc: 0.9449 - val_loss: 0.1052 - val_acc: 0.9687
    Epoch 12/20
    60000/60000 [==============================] - 2s - loss: 0.1735 - acc: 0.9463 - val_loss: 0.1022 - val_acc: 0.9707
    Epoch 13/20
    60000/60000 [==============================] - 2s - loss: 0.1709 - acc: 0.9473 - val_loss: 0.1052 - val_acc: 0.9688
    Epoch 14/20
    60000/60000 [==============================] - 2s - loss: 0.1693 - acc: 0.9476 - val_loss: 0.1044 - val_acc: 0.9698
    Epoch 15/20
    60000/60000 [==============================] - 2s - loss: 0.1710 - acc: 0.9477 - val_loss: 0.1029 - val_acc: 0.9716
    Epoch 16/20
    60000/60000 [==============================] - 2s - loss: 0.1653 - acc: 0.9483 - val_loss: 0.1048 - val_acc: 0.9694
    Epoch 17/20
    60000/60000 [==============================] - 2s - loss: 0.1644 - acc: 0.9483 - val_loss: 0.1013 - val_acc: 0.9705
    Epoch 18/20
    60000/60000 [==============================] - 2s - loss: 0.1648 - acc: 0.9494 - val_loss: 0.1007 - val_acc: 0.9724
    Epoch 19/20
    60000/60000 [==============================] - 2s - loss: 0.1627 - acc: 0.9499 - val_loss: 0.0996 - val_acc: 0.9706
    Epoch 20/20
    60000/60000 [==============================] - 2s - loss: 0.1594 - acc: 0.9492 - val_loss: 0.0990 - val_acc: 0.9707
    Train on 60000 samples, validate on 10000 samples
    Epoch 1/20
    60000/60000 [==============================] - 3s - loss: 1.4475 - acc: 0.5121 - val_loss: 0.6030 - val_acc: 0.8638
    Epoch 2/20
    60000/60000 [==============================] - 2s - loss: 1.1656 - acc: 0.6221 - val_loss: 0.5182 - val_acc: 0.8860
    Epoch 3/20
    60000/60000 [==============================] - 2s - loss: 1.1180 - acc: 0.6479 - val_loss: 0.4765 - val_acc: 0.8865
    Epoch 4/20
    60000/60000 [==============================] - 2s - loss: 1.1087 - acc: 0.6552 - val_loss: 0.4528 - val_acc: 0.8871
    Epoch 5/20
    60000/60000 [==============================] - 2s - loss: 1.1027 - acc: 0.6573 - val_loss: 0.4557 - val_acc: 0.8910
    Epoch 6/20
    60000/60000 [==============================] - 2s - loss: 1.1002 - acc: 0.6637 - val_loss: 0.4420 - val_acc: 0.8874
    Epoch 7/20
    60000/60000 [==============================] - 2s - loss: 1.1031 - acc: 0.6675 - val_loss: 0.4420 - val_acc: 0.8865
    Epoch 8/20
    60000/60000 [==============================] - 2s - loss: 1.1055 - acc: 0.6686 - val_loss: 0.4442 - val_acc: 0.8918
    Epoch 9/20
    60000/60000 [==============================] - 2s - loss: 1.1086 - acc: 0.6740 - val_loss: 0.4413 - val_acc: 0.8915
    Epoch 10/20
    60000/60000 [==============================] - 2s - loss: 1.1019 - acc: 0.6777 - val_loss: 0.4364 - val_acc: 0.8939
    Epoch 11/20
    60000/60000 [==============================] - 2s - loss: 1.1054 - acc: 0.6758 - val_loss: 0.4101 - val_acc: 0.8938
    Epoch 12/20
    60000/60000 [==============================] - 2s - loss: 1.1087 - acc: 0.6766 - val_loss: 0.4216 - val_acc: 0.8978
    Epoch 13/20
    60000/60000 [==============================] - 2s - loss: 1.1184 - acc: 0.6781 - val_loss: 0.4252 - val_acc: 0.8973
    Epoch 14/20
    60000/60000 [==============================] - 2s - loss: 1.1199 - acc: 0.6770 - val_loss: 0.4265 - val_acc: 0.8936
    Epoch 15/20
    60000/60000 [==============================] - 3s - loss: 1.1307 - acc: 0.6783 - val_loss: 0.4425 - val_acc: 0.8929
    Epoch 16/20
    60000/60000 [==============================] - 2s - loss: 1.1416 - acc: 0.6777 - val_loss: 0.4446 - val_acc: 0.8958
    Epoch 17/20
    60000/60000 [==============================] - 2s - loss: 1.1281 - acc: 0.6794 - val_loss: 0.4270 - val_acc: 0.8985
    Epoch 18/20
    60000/60000 [==============================] - 2s - loss: 1.1334 - acc: 0.6803 - val_loss: 0.4321 - val_acc: 0.8989
    Epoch 19/20
    60000/60000 [==============================] - 2s - loss: 1.1391 - acc: 0.6803 - val_loss: 0.4113 - val_acc: 0.8963
    Epoch 20/20
    60000/60000 [==============================] - 2s - loss: 1.1381 - acc: 0.6801 - val_loss: 0.4134 - val_acc: 0.8959


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


