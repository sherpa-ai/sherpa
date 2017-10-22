static_text = """
# SHERPA
Welcome to SHERPA. Our goal is to speed up machine learning research by
providing a tool for hyperparameter tuning.

### Dependencies
+ Pandas 0.19.2

In order to get SHERPA running clone the repository from GitLab by
calling ```git clone git@gitlab.ics.uci.edu:uci-igb/sherpa.git``` from the
command line and adding the directory to the Python path (e.g.
```export PYTHONPATH=$PYTHONPATH:/user/local/sherpa/```). In order to get the
necessary dependencies you can run ```python setup.py``` from the SHERPA folder.

## Getting Started
For a first step navigate to ```sherpa/examples``` and run
```python optimize_bianchini.py```. In this example SHERPA calls the script
```bianchini.py``` which trains a shallow neural network on the Bianchini function.

## Optimizing a CNN for MNIST
The next example runs a small hyperparameter optimization on a Convolutional
Neural Network trained on the MNIST dataset. The CNN is implemented in Keras so
be sure to run ```pip install keras``` in case you don't have Keras installed.
If you have a GPU machine available this will speed up the running time of the
optimization significantly. Be sure to use
```ssh -L 16006:127.0.0.1:6006 username@hostname``` when SSHing into a remote
server so you can use the visualization. Now go ahead and run
```python sherpa_mnist.py```.

### Visualizing Results
After running the command you SHERPA will display output in the terminal.
Among this it will display the address of the dashboard. If you are running 
SHERPA on your laptop or desktop you can go to ```0.0.0.0:6006``` in your
webbrowser. If you are using SSH you can go to ```0.0.0.0:16006``` (that is the
local port that we forwarded ```6006``` to). At the address you will see a
parallel coordinates plot and a table. The table shows the results so far. Hit
refresh to get the latest results. Each row represents one trial consisting of
a hyperparameter configuration and at least one metric. The plot is linked to the table. If you
hover over any of the rows in the table the corresponding line will highlight
in the plot. You can brush over the loss axis in the plot to select only the
models with the lowest loss. You can then check if you see patterns on the other
coordinates. Once the optimization finished the webserver shuts down. To view
the results again go to the folder titled ```output_[datetime]``` that has been
created in the examples folder. Now call ```python -m SimpleHTTPServer 6006```
from the command line (or ```python -m http.server 6006``` for Python 3).

<img src="./parcords.png" alt="Parallel Coordinates Plot">

### Training in Parallel
```--max_concurrent [no of processes]``` where
```[no of processes]``` is the number of processes you want to run in parallel.
If you have GPUs this should be set to the number of available GPUs on the
machine. If you have a machine with many CPUs you can set this to the number of
CPUs you intend to use.

### SGE
In addition to the local scheduler that you used above SHERPA also supports
Sun Grid Engine (SGE). This can be useful if you want to use GPUs across
multiple machines. You can call  ```python sherpa_mnist.py --sge``` and pass
the SGE project name ```-P```, the queue name ```-q``` and the resources
```-l``` as arguments.

#### Baldi Group
The default is set to submit to the Arcus 5 to 9 machines and none of the flags
need to be set. However, be sure to submit the script from
```nimbus.ics.uci.edu```.

## Authoring your own Optimization

### Local vs. SGE


## API

### Supported Algorithms

### Writing your own algorithms

"""

with open('README.md', 'w') as f:
    f.write(static_text)