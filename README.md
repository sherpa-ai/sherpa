
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

## Environment
You should have an environment-profile that sets path variables and potentially loads a Python Virtual environment. All variable settings above should go into that profile. Note that an SGE job will not load your `.bashrc` so all necessary settings need to be in your profile.

## SGE
SGE required submit options. In Sherpa, those are defined as a string via the `submit_options` argument in the scheduler. To run jobs on the Arcus machines, typical submit options would be: 
```-N myScript -P arcus.p -q arcus.q -l hostname='(arcus-1|arcus-2|arcus-3)'```.
The `-N` option defines the name. To run from Arcus 5 to 9 you would set `-q arcus-ubuntu.q` and `hostname` with the relevant machines you want to run on. The SHERPA runner script can run from any Arcus machine.

## Example
You can run an example by doing:
```
cd /your/path/sherpa/examples/bianchini/
python runner.py --env <path/to/your/environment>
```


