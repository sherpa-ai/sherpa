import os
import sherpa
from sherpa.hyperparameters import DistributionHyperparameter as Hyperparameter
from sherpa.scheduler import LocalScheduler,SGEScheduler

def run_example():
    '''
    Run parallel Sherpa optimization over a set of discrete hp combinations.
    '''
    # Iterate algorithm accepts dictionary containing lists of possible values. 
    hp_space = {'act':['tanh', 'relu'],#, 'relu'],
                'lrinit': [0.1, 0.01],
                'momentum':[0.0],
                'lrdecay':[0.0],
                'arch': [[20,5], [20, 10], [10,10,10]],
                'epochs': [4],
                }
    #alg = sherpa.algorithms.Iterate(epochs=2, hp_ranges=hp_space)
    alg = sherpa.algorithms.LocalSearch(hp_ranges=hp_space, hp_init=None)
    f   = './bianchini.py'
    dir = './output' # All files written to here.
    env = '/home/pjsadows/profiles/auto.profile' # Script specifying environment variables.
    opt = '-N demo -P arcus.p -q arcus-ubuntu.q -q arcus.q -l hostname=\'(arcus-2)\'' # SGE options.
    sched = LocalScheduler() # Run on local machine without SGE.
    #sched = SGEScheduler(environment=env, submit_options=opt)
    rval = sherpa.optimize(filename=f, algorithm=alg, dir=dir, overwrite=True, scheduler=sched, max_concurrent=4)
    print()
    print('Best results:')
    print(rval)

def run_example_advanced():
    ''' 
    Run Sherpa hyperparameter optimization.
    User may want to run this as a separate file.
    '''
    # Hyperparameter space. 
    hp_space = [
                 Hyperparameter(name='lrinit', distribution='choice', dist_args=[0.1, 0.01, 0.001]),
                 Hyperparameter(name='lrdecay', distribution='choice', dist_args=[0.0]),
                 Hyperparameter(name='momentum', distribution='choice', dist_args=[0.0, 0.5, 0.9]),
                 Hyperparameter(name='act', distribution='choice', dist_args=['tanh','relu']),
                ]
    
    # Specify how initial hp combinations are sampled.
    sampler =  sherpa.samplers.LatinHypercube # Or sherpa.samplers.RandomGenerator
     
    # Algorithm used for optimization.
    raise NotImplementedError('These algorithms need to be updated in sherpa.algorithms.')
    #alg  = sherpa.algorithms.Hyperhack(samples=4, epochs_per_stage=2, stages=4, survival=0.5, sampler=sampler, hp_ranges=hp_space)
    #alg  = sherpa.algorithms.RandomSearch(samples=100, epochs=1, hp_ranges=hp_ranges, max_concurrent=10)
    f   = './bianchini.py'
    dir = './output' # All files written to here.
    env = '/home/pjsadows/profiles/auto.profile' # Script specifying environment variables.
    opt = '-N myexample -P arcus.p -q arcus-ubuntu.q -q arcus.q -l hostname=\'(arcus-1|arcus-2)\'' # SGE options.
    sched = SGEScheduler(environment=env, submit_options=opt)
    rval = sherpa.optimize(filename=f, algorithm=alg, dir=dir, overwrite=True, scheduler=sched, max_concurrent=4)
    print()
    print('Best results:')
    print(rval)

if __name__=='__main__':
    run_example() # Sherpa optimization.

