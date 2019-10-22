"""
SHERPA is a Python library for hyperparameter tuning of machine learning models.
Copyright (C) 2018  Lars Hertel, Peter Sadowski, and Julian Collado.

This file is part of SHERPA.

SHERPA is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

SHERPA is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with SHERPA.  If not, see <http://www.gnu.org/licenses/>.
"""
import pytest
import sherpa
import sherpa.core
import sherpa.schedulers
import sherpa.database
import pandas
import collections
try:
    import unittest.mock as mock
except ImportError:
    import mock
from testing_utils import *


def test_trial():
    p = {'a': 1, 'b': 2}
    t = sherpa.Trial(1, p)
    assert t.id == 1
    assert t.parameters == p


def test_parameters():
    c, cl, d, dl, ch = get_test_parameters()

    assert c.name == 'a'
    assert all(1 <= c.sample() <= 2 for _ in range(10))
    assert all(1 <= cl.sample() <= 2 for _ in range(10))
    assert all(1 <= d.sample() <= 10 for _ in range(10))
    assert all(1 <= dl.sample() <= 10 for _ in range(10))
    assert all(ch.sample() in [1, 10] for _ in range(10))





def test_study_add_observation_no_iteration():
    s = get_mock_study()
    t = s.get_suggestion()
    s.add_observation(trial=t, objective=0.1, context={'other_metrics': 0.2})
    assert list(s.results.T.to_dict().values()) == [{'Trial-ID': 1, 'Objective': 0.1, 'other_metrics': 0.2, 'a':1 , 'b':2, 'Status': sherpa.TrialStatus.INTERMEDIATE, 'Iteration': 1}]


def test_study_add_observation_same_iteration_twice():
    s = get_mock_study()
    t = s.get_suggestion()
    s.add_observation(trial=t, objective=0.1, context={'other_metrics': 0.2}, iteration=1)
    with pytest.raises(ValueError):
        s.add_observation(trial=t, objective=0.1, context={'other_metrics': 0.2}, iteration=1)


def test_study_add_observation_with_iteration():
    s = get_mock_study()
    t = s.get_suggestion()
    s.add_observation(trial=t, objective=0.1, context={'other_metrics': 0.2},
                      iteration=1)
    s.add_observation(trial=t, objective=0.01, context={'other_metrics': 0.02},
                      iteration=2)
    expected_df = pandas.DataFrame(collections.OrderedDict(
        [('Trial-ID', [1, 1]),
         ('Status', ['INTERMEDIATE', 'INTERMEDIATE']),
         ('Iteration', [1, 2]),
         ('a', [1, 1]),
         ('b', [2, 2]),
         ('Objective', [0.1, 0.01]),
         ('other_metrics', [0.2, 0.02])]
    ))
    assert s.results.equals(expected_df)


def test_study_add_observation_invalid_trial():
    s = get_mock_study()
    with pytest.raises(ValueError):
        s.add_observation(trial=sherpa.Trial(id=1, parameters={'abcd': 1}), objective=0.1, context={'other_metrics': 0.2}, iteration=1)


def test_study_finalize():
    s = get_mock_study()

    t = s.get_suggestion()
    assert t.id == 1
    assert t.parameters == {'a': 1, 'b': 2}
    s.algorithm.get_suggestion.assert_called_with(s.parameters, s.results,
                                                  s.lower_is_better)

    s.add_observation(trial=t, iteration=1, objective=0.1,
                      context={'other_metrics': 0.2})
    s.add_observation(trial=t, iteration=2, objective=0.01,
                      context={'other_metrics': 0.02})
    s.finalize(trial=t, status='COMPLETED')

    expected_df = pandas.DataFrame(collections.OrderedDict(
        [('Trial-ID', [1, 1, 1]),
         ('Status', ['INTERMEDIATE', 'INTERMEDIATE', 'COMPLETED']),
         ('Iteration', [1, 2, 2]),
         ('a', [1, 1, 1]),
         ('b', [2, 2, 2]),
         ('Objective', [0.1, 0.01, 0.01]),
         ('other_metrics', [0.2, 0.02, 0.02])]
    ))

    assert s.results.equals(expected_df)
