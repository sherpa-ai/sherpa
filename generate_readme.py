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

## Installation
Clone into ```/your/path/``` from GitLab:
```
cd /your/path/
git clone git@gitlab.ics.uci.edu:uci-igb/sherpa.git
```

Add SHERPA and GPU_LOCK to Python-path in your profile:
```
export PYTHONPATH=$PYTHONPATH:/your/path/sherpa/
export PYTHONPATH=$PYTHONPATH:/extra/pjsadows0/libs/shared/gpu_lock/
```

Add MongoDB, DRMAA and SGE to your profile:
```
source /auto/igb-libs/linux/centos/6.x/x86_64/profiles/general
export DRMAA_LIBRARY_PATH=/opt/sge/lib/lx-amd64/libdrmaa.so
module load sge
```

Install dependencies:
```
cd /your/path/sherpa
pip install -e .
```

or

```
pip install pandas
pip install pymongo
pip install numpy
pip install scipy
pip install sklearn
pip install flask
pip install drmaa
pip install enum34  # if on < Python 3.4
```

## Environment
You should have an environment-profile that sets path variables and potentially loads a Python Virtual environment. All variable settings above should go into that profile. Note that an SGE job will not load your `.bashrc` so all necessary settings need to be in your profile.

## SGE
SGE required submit options. In Sherpa, those are defined as a string via the `submit_options` argument in the scheduler. To run jobs on the Arcus machines, typical submit options would be: 
```-N myScript -P arcus.p -q arcus.q -l hostname=\'(arcus-1|arcus-2|arcus-3)\'```.
The `-N` option defines the name. To run from Arcus 5 to 9 you would set `-q arcus-ubuntu.q` and `hostname` with the relevant machines you want to run on. The SHERPA runner script can run from any Arcus machine.

## Example
You can run an example by doing:
```
cd /your/path/sherpa/examples/bianchini/
python runner.py --env <path/to/your/environment>
```


"""

# text += "### Parameters"
# text += process_function_docstring(sherpa.Parameter.__doc__)
# text += "\n"
#
# text += "### Algorithm"
# text += process_function_docstring(sherpa.algorithms.Algorithm.__doc__)
# text += "\n"
#
# text += "### Stopping Rules"
# text += process_function_docstring(sherpa.algorithms.StoppingRule.__doc__)
# text += "\n"
#
# text += "### Combining these into a Study"
# text += process_function_docstring(sherpa.Study.__doc__)
# text += "\n"
#
# text += "### Scheduler"
# text += process_function_docstring(sherpa.schedulers.Scheduler.__doc__)
# text += "\n"
#
# text += "### Putting it all together"
# text += process_function_docstring(sherpa.optimize.__doc__)
# text += "\n"


with open('README.md', 'w') as f:
    f.write(text)
