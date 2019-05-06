import collections
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sherpa.algorithms import Algorithm
from sherpa.core import TrialStatus, AlgorithmState
import pandas
import math


class SequentialTesting(Algorithm):
    """
    Implements repeated trials based on a
    group sequential testing design.

    Args:
        algorithm (sherpa.algorithms.Algorithm): a Sherpa
            algorithm to suggest parameter configurations.
        K (int): the number of candidate configurations to
            evaluate.
        n (iterable): cumulative sample sizes at which to conduct
            interim analyses.
        P (float): between 0 and 1, defines the sequential
            testing boundary as in
            @article{kittelson1999unifying,
              title={A unifying family of group sequential test designs},
              author={Kittelson, John M and Emerson, Scott S},
              journal={Biometrics},
              volume={55},
              number={3},
              pages={874--882},
              year={1999},
              publisher={Wiley Online Library}
            }
    """

    def __init__(self, algorithm, K, n=(3, 6, 9), P=0.5, alpha=0.05, verbose=False):
        self.algorithm = algorithm
        self.K = {1: K}
        self.k = 0
        self.t = 1
        self.n = n
        self.verbose = verbose
        if len(n) > 1:
            self.dn = {i + 1: t - s for i, (s, t) in
                       enumerate(zip([0] + list(self.n), list(self.n)))}
        else:
            self.dn = {1: n}
        self.P = P
        self.T = len(n)
        self.config_stack = []
        self.configs_by_stage = collections.defaultdict(list)
        assert list(self.n) == [3, 6,
                                9] and P == 0.5 and alpha == 0.05, "Currently only n=(3,6,9), P=0.5, and alpha=0.05 is supported"
        self.alpha_dash = [0.023175014834345, 0.023175014834345,
                           0.023175014834345]
        self.parameter_types = {}
        # reading configs from results they may
        # have the wrong type (e.g. float instead of int). This is a fix to
        # deal with this for now

    def get_suggestion(self, parameters, results=None, lower_is_better=True):
        """
        Variables:
        Stages: T
        Stage: t=1,2,3...
        Config in stage: K={1: K, 2: tilde{k}, 3: tilde{k}, ...}
        Config counter within stage: k=1,2,3...
        Sample sizes for each stage: dn={1: n_stage_1, 2: n_stage_2, ...}
        Repeats to evaluate: config_stack=[config1, config2, ...]
        Configurations by stage: {1: all the configs stage 1, 2: all configs stage 2, ...}
        """
        if len(self.config_stack) == 0:
            if self.k == self.K[self.t]:
                # end of the stage
                if self.t == self.T or self.K[self.t] == 1:
                    # end of algorithm
                    return AlgorithmState.DONE
                else:
                    if self._is_stage_done(results, stage=self.t,
                                           num_trials_for_stage=self.K[self.t] *
                                                   self.dn[self.t]):
                        # if all trials finished, do testing
                        # and append new trials

                        self.configs_by_stage[
                            self.t + 1] = self._get_best_configs(parameters,
                                                                 results,
                                                                 configs=
                                                                 self.configs_by_stage[
                                                                     self.t],
                                                                 lower_is_better=lower_is_better,
                                                                 alpha=
                                                                 self.alpha_dash[
                                                                     self.t])
                        self.t += 1
                        self.K[self.t] = len(self.configs_by_stage[self.t])
                        self.k = 1
                        self.config_stack = [self.configs_by_stage[self.t][
                                                 self.k - 1]] * self.dn[self.t]
                    else:
                        # otherwise wait for trials to finish
                        if self.verbose:
                            print(AlgorithmState.WAIT)
                        return AlgorithmState.WAIT

            else:
                self.k += 1
                if self.t == 1:
                    new_config = self.algorithm.get_suggestion(parameters,
                                                               results,
                                                               lower_is_better)
                    if len(self.parameter_types) == 0:
                        self.parameter_types = {p.name: p.type
                                                for p in parameters}
                    self.configs_by_stage[self.t].append(new_config)
                    self.config_stack = [new_config] * self.dn[self.t]
                else:
                    self.config_stack = [self.configs_by_stage[self.t][
                                             self.k - 1]] * self.dn[self.t]

        return dict({pname: self.parameter_types[pname](pvalue)
                     for pname, pvalue in self.config_stack.pop().items()},
                    **{'stage': self.t})

    @staticmethod
    def _is_stage_done(results, stage, num_trials_for_stage):
        """
        Checks whether all trials from the given stage have completed.
        """
        completed = results[results.Status == TrialStatus.COMPLETED]
        this_stage = completed[completed.stage == stage]
        return len(this_stage) == num_trials_for_stage

    def _get_best_configs(self, parameters, results, configs, lower_is_better, alpha=0.05):
        """
        Implements the testing procedure itself and returns the reduced set
        of parameter configurations.
        """
        if lower_is_better:
            df = self._prep_df_for_linreg(parameters, results, configs)
        else:
            results.Objective = -1*results.Objective
            df = self._prep_df_for_linreg(parameters, results, configs)
        print(df)
        l = 1
        h = df.Rank.max()
        p = h
        while l != h:
            lm = ols('Objective ~ C(Rank)', data=df.loc[df.Rank <= p, :]).fit()

            pval = sm.stats.anova_lm(lm, typ=2).loc[:, "PR(>F)"].ix["C(Rank)"]
            reject = pval < alpha
            if reject:
                h = p - 1
            else:
                l = p
            p = math.ceil((l + h) / 2)

        return df.loc[df.Rank <= p, :].loc[:,
               [p.name for p in parameters]].drop_duplicates().to_dict(
            'records')

    @staticmethod
    def _prep_df_for_linreg(parameters, results, configs):
        """
        Filter results corresponding to parameter configurations in `configs`
        argument and create a grouping variable Rank that corresponds to
        the rank of each group in terms of the average objective.

        Args:
            parameters (List[sherpa.Parameter]): list of parameters
            results (pandas.DataFrame): current results
            configs (List[Dict[Str, Any]]): parameter configurations that
                correspond to groups.
        """
        param_names = [p.name for p in parameters]
        completed = results[results.Status == TrialStatus.COMPLETED]
        config_df = pandas.DataFrame(configs)
        filtered_configs = completed.merge(config_df, how='inner',
                                           on=param_names)
        group_ranks = filtered_configs.groupby(param_names).agg('mean') \
                          .loc[:, ['Objective']].rank(axis=0, method='dense') \
            .astype({'Objective': int}) \
            .rename({'Objective': 'Rank'}, axis=1) \
            .reset_index() \
            .merge(filtered_configs)
        return group_ranks

