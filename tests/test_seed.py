import sherpa
from testing_utils import get_test_parameters

def test_rng():
    sherpa.rng.randn()
    sherpa.rng.seed(1234)
    a = sherpa.rng.randn()
    sherpa.rng.seed(1234)
    b = sherpa.rng.randn()
    assert a ==b

def test_random_search():
    algorithm = sherpa.algorithms.RandomSearch()
    sherpa.rng.seed(1234)
    suggestion1 = algorithm.get_suggestion(parameters=get_test_parameters())
    sherpa.rng.seed(1234)
    suggestion2 = algorithm.get_suggestion(parameters=get_test_parameters())
    assert suggestion1 == suggestion2