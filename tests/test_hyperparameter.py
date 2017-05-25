from __future__ import print_function
from sherpa.core import GrowingHyperparameter
from sherpa.hparam_generators import RandomGenerator

def test_growing_hyperparameter():
    hparam = GrowingHyperparameter(name='num_units', choices=[5, 10, 20],
                                   start_value=100)
    rg = RandomGenerator([hparam])
    print(rg.next())
    print(hparam.distr_args)
    hparam.grow(value=5, amount=100)
    hparam.grow(value=20, amount=-100)
    print(hparam.distr_args)
    assert all(rg.next()['num_units'] != 20 for i in range(1000))
    hparam.grow(value=20, amount=-100)
    print(hparam.distr_args)

if __name__ == '__main__':
    test_growing_hyperparameter()