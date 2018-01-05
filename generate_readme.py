import re
import sherpa


# from keras
def process_function_docstring(docstring):
    docstring = re.sub(r'\n    # (.*)\n',
                       r'\n    __\1__\n\n',
                       docstring)
    docstring = re.sub(r'    ([^\s\\\(]+) (.*):(.*)\n',
                       r'    - __\1__ _\2_:\3\n',
                       docstring)

    docstring = docstring.replace('    ' * 6, '\t\t')
    docstring = docstring.replace('    ' * 4, '\t')
    docstring = docstring.replace('    ', '')
    return docstring


text = """
# SHERPA

Welcome to SHERPA - a hyperparameter tuning framework for machine learning.
In order to get SHERPA running clone the repository from GitLab by
calling ```git clone git@gitlab.ics.uci.edu:uci-igb/sherpa.git``` from the
command line and adding the directory to the Python path (e.g.
```export PYTHONPATH=$PYTHONPATH:/user/local/sherpa/```). In order to get the
necessary dependencies you can run ```python setup.py install``` from the SHERPA folder.

### Optional Dependencies
+ Drmaa 0.7.8 (for SGE)
+ Keras (for examples)
+ GPU Lock (for examples and recommended for SGE)

## Running Sherpa

"""

text += "### Parameters"
text += process_function_docstring(sherpa.Parameter.__doc__)
text += "\n"

text += "### Algorithm"
text += process_function_docstring(sherpa.algorithms.Algorithm.__doc__)
text += "\n"

text += "### Stopping Rules"
text += process_function_docstring(sherpa.algorithms.StoppingRule.__doc__)
text += "\n"

text += "### Combining these into a Study"
text += process_function_docstring(sherpa.Study.__doc__)
text += "\n"

text += "### Scheduler"
text += process_function_docstring(sherpa.schedulers.Scheduler.__doc__)
text += "\n"

text += "### Putting it all together"
text += process_function_docstring(sherpa.optimize.__doc__)
text += "\n"


with open('README.md', 'w') as f:
    f.write(text)
