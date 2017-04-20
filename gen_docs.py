import re
from hobbit.algorithms import Hyperband
from hobbit import Hyperparameter

def process_class_docstring(docstring):
    docstring = re.sub(r'\n    # (.*)\n',
                       r'\n    __\1__\n\n',
                       docstring)

    docstring = re.sub(r'    ([^\s\\]+):(.*)\n',
                       r'    - __\1__:\2\n',
                       docstring)

    docstring = docstring.replace('    ' * 5, '\t\t')
    docstring = docstring.replace('    ' * 3, '\t')
    docstring = docstring.replace('    ', '')
    return docstring

open('./hyperband.md', 'w').write(process_class_docstring(Hyperband.__doc__))
open('./hyperparameter.md', 'w').write(process_class_docstring(
    Hyperparameter.__doc__))