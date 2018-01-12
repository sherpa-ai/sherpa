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
python setup.py install --parallel
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
