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
from sherpa.algorithms import Algorithm, RandomSearch
import numpy
import pandas
import collections
from sherpa.core import TrialStatus, AlgorithmState


class SuccessiveHalving(Algorithm):
    """
    Asynchronous Successive Halving as described in:
    @article{li2018massively,
      title={Massively parallel hyperparameter tuning},
      author={Li, Liam and Jamieson, Kevin and Rostamizadeh, Afshin and Gonina, Ekaterina and Hardt, Moritz and Recht, Benjamin and Talwalkar, Ameet},
      journal={arXiv preprint arXiv:1810.05934},
      year={2018}
    }

    Args:
        r (int): minimum resource
        R (int): maximum resource
        eta (int): reduction factor
        s (int): minimum early-stopping rate
        max_finished_configs (int): stop once max_finished_configs models have
            been trained to completion.

    """
    def __init__(self, r, R, eta, s, max_finished_configs=None):
        self.eta = eta
        self.r = r
        self.R = R
        self.s = s
        self.rs = RandomSearch()
        self.number_of_rungs = (numpy.floor(
            numpy.log(R/r) / numpy.log(eta)) - s).astype('int')
        self.config_counter = 1
        self.promoted_trials = [set() for _ in range(self.number_of_rungs)]
        self.not_promoted_trials = [set() for _ in range(self.number_of_rungs+1)]
        self.max_finished_configs = max_finished_configs

    def get_suggestion(self, parameters, results, lower_is_better):
        if (self.max_finished_configs and
                    self.max_finished_configs <= len(self.not_promoted_trials[self.number_of_rungs])):
            return AlgorithmState.DONE

        config, k = self.get_job(parameters, results, lower_is_better)
        config['resource'] = self.r * self.eta ** (self.s + k)
        config['rung'] = k
        config['load_from'] = config.get('save_to', '')
        config['save_to'] = str(self.config_counter)
        self.not_promoted_trials[k].add(config['save_to'])
        self.config_counter += 1
        return config

    def get_job(self, parameters, results, lower_is_better):
        """
        Check to see if there is a promotable configuration. Otherwise,
        return a new configuration.
        """
        for k in reversed(range(self.number_of_rungs)):
            candidates = self.top_n(parameters,
                                    results,
                                    lower_is_better,
                                    rung=k,
                                    eta=self.eta)
            # print("RUNG", k, "CANDIDATES\n", candidates)
            promotable = candidates[~candidates.save_to.isin(
                self.promoted_trials[k])].to_dict('records')

            if len(promotable) > 0:
                # What would the proportion of promotions be after the promotion?
                n_promoted = len(self.promoted_trials[k])
                n_not_promoted = len(self.not_promoted_trials[k])
                proportion_after_promotion = (n_promoted + 1) / (n_not_promoted + n_promoted)
                if proportion_after_promotion <= 1/self.eta:
                    print("PROMOTE")
                    self.promote(rung=k, trial=promotable[0])
                    return promotable[0], k+1
        else:
            new_config = self.rs.get_suggestion(parameters=parameters)
            return new_config, 0

    @staticmethod
    def top_n(parameters, results, lower_is_better, rung, eta):
        """
        If there are >=m*eta configs in rung k, return the m best ones,
        otherwise return []
        """
        if len(results) == 0:
            return pandas.DataFrame({'save_to': []})
        columns = [p.name for p in parameters] + ['save_to']
        rung_results = results.query("Status == '{}'".format(
            TrialStatus.COMPLETED)) \
            .query("rung == {}".format(rung))

        n = len(rung_results) // eta
        top_n = rung_results.sort_values(by="Objective",
                                         ascending=lower_is_better) \
                            .iloc[0:n, :] \
                            .loc[:, columns] \

        return top_n

    def promote(self, rung, trial):
        """
        Stores that the trial in the given rung has been promoted.
        """
        self.not_promoted_trials[rung].remove(trial['save_to'])
        self.promoted_trials[rung].add(trial['save_to'])
