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
import os
import random
import numpy
import logging
import sherpa
import pandas
import scipy.stats
import scipy.optimize
import sklearn.gaussian_process
from .core import Choice, Continuous, Discrete, Ordinal
import sklearn.model_selection
from sklearn import preprocessing
import warnings
import collections


logging.basicConfig(level=logging.DEBUG)
alglogger = logging.getLogger(__name__)


class Algorithm(object):
    """
    Abstract algorithm that generates new set of parameters.
    """
    def get_suggestion(self, parameters, results, lower_is_better):
        """
        Returns a suggestion for parameter values.

        Args:
            parameters (list[sherpa.Parameter]): the parameters.
            results (pandas.DataFrame): all results so far.
            lower_is_better (bool): whether lower objective values are better.

        Returns:
            dict: parameter values.
        """
        raise NotImplementedError("Algorithm class is not usable itself.")

    def load(self, num_trials):
        """
        Reinstantiates the algorithm when loaded.

        Args:
            num_trials (int): number of trials in study so far.
        """
        pass


class RandomSearch(Algorithm):
    """
    Regular Random Search.

    Args:
        max_num_trials (int): number of trials, otherwise runs indefinitely.
    """
    def __init__(self, max_num_trials=None):
        self.max_num_trials = max_num_trials
        self.count = 0

    def get_suggestion(self, parameters, results=None, lower_is_better=True):
        if self.max_num_trials and self.count >= self.max_num_trials:
            return None
        else:
            self.count += 1
            return {p.name: p.sample() for p in parameters}

class Iterate(Algorithm):
    """
    Iterate over a set of fully-specified hyperparameter combinations.
    
    Args:
        hp_iter (list): list of fully-specified hyperparameter dicts. 
    """
    def __init__(self, hp_iter):
        self.hp_iter = hp_iter
        self.count = 0
        
        # Make sure all hyperparameter values are specified.
        parameters = self.get_parameters()
    
    def get_suggestion(self, parameters, results=None, lower_is_better=True):
        if self.count >= len(self.hp_iter):
            # No more combinations to try.
            return None
        else:
            hp = self.hp_iter[self.count]
            self.count += 1
            return hp

    def load(self, num_trials):
        self.count = num_trials
        
    def get_parameters(self):
        """
        Computes list of parameter objects from list of hyperparameter
        combinations, which is needed for initializing a Study.
        
        Returns:
            list: List of Parameter objects.
        """
        parameters = []
        keys = self.hp_iter[0].keys()
        for pname in keys:
            # Get unique values of this (possibly unhashable) parameter.
            prange = []
            for i,hp in enumerate(self.hp_iter):
                if pname not in hp:
                    raise Exception('Parameter {pname} not found in list item {i}.')
                value = hp[pname]
                if value not in prange:
                    prange.append(value)
            p = sherpa.Parameter.from_dict({'name': pname,
                                     'type': 'choice',
                                     'range': prange})
            parameters.append(p)
        return parameters
    
class GridSearch(Algorithm):
    """
    Regular Grid Search. Expects ``Choice`` or ``Ordinal`` parameters.

    For continuous and discrete parameters grid points are picked within the
    range. For example, a continuous parameter with range [1, 2] with two grid
    points would have points 1 1/3 and 1 2/3. For three points, 1 1/4, 1 1/2,
    and 1 3/4.
    
    Example:
    ::

        hp_space = {'act': ['tanh', 'relu'],
                    'lrinit': [0.1, 0.01],
                    }
        parameters = sherpa.Parameter.grid(hp_space)
        alg = sherpa.algorithms.GridSearch()

    Args:
        num_grid_points (int): number of grid points for continuous / discrete.

    """
    def __init__(self, num_grid_points=2):
        self.count = 0
        self.grid = None
        self.num_grid_points = num_grid_points

    def get_suggestion(self, parameters, results=None, lower_is_better=True):
        if self.count == 0:
            param_dict = self._get_param_dict(parameters)
            self.grid = list(sklearn.model_selection.ParameterGrid(param_dict))
        if self.count == len(self.grid):
            return None
        else:
            params = self.grid[self.count]
            self.count += 1
            return params

    def _get_param_dict(self, parameters):
        param_dict = {}
        for p in parameters:
            if isinstance(p, Continuous) or isinstance(p, Discrete):
                values = []
                for i in range(self.num_grid_points):
                    if p.scale == 'log':
                        v = numpy.log10(p.range[1]) - numpy.log10(p.range[0])
                        v *= (i + 1) / (self.num_grid_points + 1)
                        v += numpy.log10(p.range[0])
                        v = 10**v
                        if isinstance(p, Discrete):
                            v = int(v)
                        values.append(v)
                    else:
                        v = p.range[1]-p.range[0]
                        v *= (i + 1)/(self.num_grid_points + 1)
                        v += p.range[0]
                        if isinstance(p, Discrete):
                            v = int(v)
                        values.append(v)
            else:
                values = p.range
            param_dict[p.name] = values
        return param_dict


class LocalSearch(Algorithm):
    """
    Local Search Algorithm.

    This algorithm expects to start with a very good hyperparameter
    configuration. It changes one hyperparameter at a time to see if better
    results can be obtained.

    Args:
        seed_configuration (dict): hyperparameter configuration to start with.
        perturbation_factors (Union[tuple,list]): continuous parameters will be
            multiplied by these.
        repeat_trials (int): number of times that identical configurations are
            repeated to test for random fluctuations.
    """
    def __init__(self, seed_configuration, perturbation_factors=(0.8, 1.2), repeat_trials=1):
        self.seed_configuration = seed_configuration
        self.count = 0
        self.submitted = []
        self.perturbation_factors = perturbation_factors
        self.next_trial = []
        self.repeat_trials = repeat_trials
        
    def get_suggestion(self, parameters, results, lower_is_better):
        if not self.next_trial:
            self.next_trial = self._get_next_trials(parameters, results,
                                                   lower_is_better)

        return self.next_trial.pop()

    def _get_next_trials(self, parameters, results, lower_is_better):
        self.count += 1
        if self.count == 1:
            self.submitted.append(self.seed_configuration)
            return [self.seed_configuration] * self.repeat_trials

        # Get best result so far
        if len(results) > 0:
            best_idx = (results.loc[:, 'Objective'].idxmin() if lower_is_better
                        else results.loc[:, 'Objective'].idxmax())
            self.seed_configuration = results.loc[
                best_idx, [p.name for p in parameters]].to_dict()

        # Randomly sample perturbations and return first that hasn't been tried
        for param in random.sample(parameters, len(parameters)):
            if isinstance(param, Choice):
                values = random.sample(param.range,
                                       len(param.range))
                for val in values:
                    new_params = self.seed_configuration.copy()
                    new_params[param.name] = val
                    if new_params not in self.submitted:
                        self.submitted.append(new_params)
                        return [new_params] * self.repeat_trials
            else:
                for incr in random.sample([True, False], 2):
                    new_params = self._perturb(candidate=self.seed_configuration.copy(),
                                               parameter=param,
                                               increase=incr)
                    if new_params not in self.submitted:
                        self.submitted.append(new_params)
                        return [new_params] * self.repeat_trials
        else:
            alglogger.info("All local perturbations have been exhausted and "
                           "no better local optimum was found.")
            return [None] * self.repeat_trials

    def _perturb(self, candidate, parameter, increase):
        """
        Randomly choose one parameter and perturb it.

        For Ordinal this is increased/decreased, for continuous/discrete this is
        times 0.8/1.2.

        Args:
            parameters (list[sherpa.core.Parameter]): parameter ranges.
            configuration (dict): a parameter configuration to be perturbed.
            param_name (str): the name of the parameter to perturb.
            increase (bool): whether to increase or decrease the parameter.

        Returns:
            dict: perturbed configuration
        """
        if isinstance(parameter, Ordinal):
            shift = +1 if increase else -1
            values = parameter.range
            newidx = values.index(candidate[parameter.name]) + shift
            newidx = numpy.clip(newidx, 0, len(values) - 1)
            candidate[parameter.name] = values[newidx]

        else:
            factor = self.perturbation_factors[1 if increase else 0]
            candidate[parameter.name] *= factor

            if isinstance(parameter, Discrete):
                candidate[parameter.name] = int(candidate[parameter.name])

            candidate[parameter.name] = numpy.clip(candidate[parameter.name],
                                               min(parameter.range),
                                               max(parameter.range))
        return candidate


class StoppingRule(object):
    """
    Abstract class to evaluate whether a trial should stop conditional on all
    results so far.
    """
    def should_trial_stop(self, trial, results, lower_is_better):
        """
        Args:
            trial (sherpa.Trial): trial to be stopped.
            results (pandas.DataFrame): all results so far.
            lower_is_better (bool): whether lower objective values are better.

        Returns:
            bool: decision.
        """
        raise NotImplementedError("StoppingRule class is not usable itself.")


class MedianStoppingRule(StoppingRule):
    """
    Median Stopping-Rule similar to Golovin et al.
    "Google Vizier: A Service for Black-Box Optimization".

    * For a Trial `t`, the best objective value is found.
    * Then the best objective value for every other trial is found.
    * Finally, the best-objective for the trial is compared to the median of
      the best-objectives of all other trials.

    If trial `t`'s best objective is worse than that median, it is
    stopped.

    If `t` has not reached the minimum iterations or there are not
    yet the requested number of comparison trials, `t` is not
    stopped. If `t` is all nan's it is stopped by default.

    Args:
        min_iterations (int): the minimum number of iterations a trial runs for
            before it is considered for stopping.
        min_trials (int): the minimum number of comparison trials needed for a
            trial to be stopped.
    """
    def __init__(self, min_iterations=0, min_trials=1):
        self.min_iterations = min_iterations
        self.min_trials = min_trials

    def should_trial_stop(self, trial, results, lower_is_better):
        """
        Args:
            trial (sherpa.Trial): trial to be stopped.
            results (pandas.DataFrame): all results so far.
            lower_is_better (bool): whether lower objective values are better.

        Returns:
            bool: decision.
        """
        if len(results) == 0:
            return False
        
        trial_rows = results.loc[results['Trial-ID'] == trial.id]
        
        max_iteration = trial_rows['Iteration'].max()
        if max_iteration < self.min_iterations:
            return False
        
        trial_obj_val = trial_rows['Objective'].min() if lower_is_better else trial_rows['Objective'].max()

        if numpy.isnan(trial_obj_val) and not trial_rows.empty:
            alglogger.debug("Value {} is NaN: {}, trial rows: {}".format(trial_obj_val, numpy.isnan(trial_obj_val), trial_rows))
            return True

        other_trial_ids = set(results['Trial-ID']) - {trial.id}
        comparison_vals = []

        for tid in other_trial_ids:
            trial_rows = results.loc[results['Trial-ID'] == tid]
            
            max_iteration = trial_rows['Iteration'].max()
            if max_iteration < self.min_iterations:
                continue

            valid_rows = trial_rows.loc[trial_rows['Iteration'] <= max_iteration]
            obj_val = valid_rows['Objective'].min() if lower_is_better else valid_rows['Objective'].max()
            comparison_vals.append(obj_val)

        if len(comparison_vals) < self.min_trials:
            return False

        if lower_is_better:
            decision = trial_obj_val > numpy.nanmedian(comparison_vals)
        else:
            decision = trial_obj_val < numpy.nanmedian(comparison_vals)

        return decision


def get_sample_results_and_params():
    """
    Call as:
    ::

        parameters, results, lower_is_better = sherpa.algorithms.get_sample_results_and_params()


    to get a sample set of parameters, results and lower_is_better variable.
    Useful for algorithm development.

    Note: losses are obtained from
    ::

        loss = param_a / float(iteration + 1) * param_b

    """
    here = os.path.abspath(os.path.dirname(__file__))
    results = pandas.read_csv(os.path.join(here, "sample_results.csv"), index_col=0)
    parameters = [Choice(name="param_a",
                         range=[1, 2, 3]),
                  Continuous(name="param_b",
                         range=[0, 1])]
    lower_is_better = True
    return parameters, results, lower_is_better


class BayesianOptimization(Algorithm):
    """
    Bayesian optimization using Gaussian Process and Expected Improvement.

    Bayesian optimization is a black-box optimization method that uses a
    probabilistic model to build a surrogate of the unknown objective function.
    It chooses points to evaluate using an acquisition function that trades off
    exploitation (e.g. high mean) and exploration (e.g. high variance).

    Args:
        num_grid_points (int): the number of grid points for continuous/discrete
            parameters. These will be evaluated first.
        max_num_trials (int): the number of trials after which the algorithm will
            stop. Defaults to ``None`` i.e. runs forever.
        acquisition_function (str): currently only ``'ei'`` for expected improvement.

    """
    def __init__(self, num_grid_points=2, max_num_trials=None, log_y=False,
                 fine_tune=True):
        self.num_grid_points = num_grid_points
        self.count = 0
        self.num_candidates = 10000
        self.num_optimized = 50
        self.max_num_trials = max_num_trials
        self.random_sampler = RandomSearch()
        self.grid_search = GridSearch(num_grid_points=num_grid_points)
        self.best_y = None
        self.epsilon = 0.
        self.lower_is_better = None
        self.gp = None
        self.log_y = log_y
        self.fine_tune = fine_tune

        self.Xcolumns = {}  # mapping: param name -> columns in X
        self.transformers = {}  # mapping: param name -> transformation object

    def get_suggestion(self, parameters, results=None,
                       lower_is_better=True):
        self.count += 1
        self.lower_is_better = lower_is_better

        if self.max_num_trials and self.count >= self.max_num_trials:
            # Algorithm finished
            return None

        seed = self.grid_search.get_suggestion(parameters=parameters)
        if seed:
            # Algorithm still in seed stage
            return seed

        completed = results.query("Status == 'COMPLETED'")
        if len(completed) == 0:
            # No completed results, return random trial
            return self.random_sampler.get_suggestion(parameters=parameters)

        # Prepare data for GP
        Xtrain = self._to_design(completed.loc[:, [p.name for p in parameters]],
                                 parameters)
        ytrain = numpy.array(completed.loc[:, 'Objective'])
        if self.log_y:
            assert all(y > 0. for y in ytrain), "Only positive objective" \
                                                "values are allowed"
            ytrain = numpy.log10(ytrain)

        self.best_y = ytrain.min() if lower_is_better else ytrain.max()

        kernel = sklearn.gaussian_process.kernels.Matern(nu=2.5, length_scale=float(2./len(ytrain)))

        self.gp = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel,
                                                                    alpha=1e-8,
                                                                    optimizer='fmin_l_bfgs_b' if len(ytrain) >= 8*len(parameters) else None,
                                                                    n_restarts_optimizer=10,
                                                                    normalize_y=True)
        self.gp.fit(Xtrain, ytrain)

        candidate_df = self._generate_candidates(parameters)
        Xcandidate = self._to_design(candidate_df, parameters)
        EI_Xcandidate = self.get_expected_improvement(Xcandidate)

        if (not all(isinstance(p, Choice) for p in parameters)) and self.fine_tune:
            # Get indexes of candidates with highest expected improvement
            best_idxs = EI_Xcandidate.argsort()[-self.num_optimized:][::-1]
            Xoptimized, EI_Xoptimized = self._maximize(Xcandidate[best_idxs],
                                                       max_function=self.get_expected_improvement)

            X_total = numpy.concatenate([Xoptimized, Xcandidate])
            EI_total = numpy.concatenate([EI_Xoptimized, EI_Xcandidate])
        else:
            X_total = Xcandidate
            EI_total = EI_Xcandidate

        # _from_design returns a dataframe so to get it into the right
        # dictionary form the row needs to be extracted
        df = self._from_design(X_total[EI_total.argmax()])

        # For debugging, so that these can be accessed from outside
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.X_total = X_total
        self.EI_total = EI_total

        return df.iloc[0].to_dict()

    def _generate_candidates(self, parameters):
        """
        Generates candidate parameter configurations via random samples.

        Args:
            parameters (list[sherpa.core.Parameter]): list of hyperparameters.

        Returns:
            pandas.DataFrame: table with hyperparameters being columns.
        """
        d = [self.random_sampler.get_suggestion(parameters)
             for _ in range(self.num_candidates)]
        return pandas.DataFrame.from_dict(d)

    class ChoiceTransformer(object):
        def __init__(self, parameter):
            self.le = preprocessing.LabelEncoder()
            self.le.fit(parameter.range)
            self.enc = preprocessing.OneHotEncoder()
            self.enc.fit(numpy.reshape(self.le.transform(parameter.range),
                                       (-1, 1)))

        def transform(self, s):
            labels = self.le.transform(numpy.array(s))
            onehot = self.enc.transform(numpy.reshape(labels, (-1, 1)))
            return onehot.toarray()

        def reverse(self, onehot):
            labels = onehot.argmax(axis=-1)
            return self.le.inverse_transform(labels)

    class ContinuousTransformer(object):
        def __init__(self, parameter):
            self.scaler = preprocessing.MinMaxScaler()
            self.log = (parameter.scale == 'log')
            if self.log:
                self.scaler.fit(numpy.log10(self._reshape(parameter.range)))
            else:
                self.scaler.fit(self._reshape(parameter.range))

        @staticmethod
        def _reshape(x):
            return numpy.array(x).astype('float').reshape((-1, 1))

        def transform(self, s):
            if self.log:
                x = self.scaler.transform(numpy.log10(self._reshape(s)))
            else:
                x = self.scaler.transform(self._reshape(s))
            return x.reshape(-1)

        def reverse(self, x):
            if self.log:
                original = 10.**self.scaler.inverse_transform(self._reshape(x))
            else:
                original = self.scaler.inverse_transform(self._reshape(x))
            return original.reshape(-1)

    class DiscreteTransformer(ContinuousTransformer):
        def __init__(self, parameter):
            super(BayesianOptimization.DiscreteTransformer, self).__init__(parameter)

        def reverse(self, x):
            return numpy.round(super(BayesianOptimization.DiscreteTransformer, self).reverse(x)).astype('int')

    def _to_design(self, df, parameters):
        """
        Turns a dataframe of parameter configurations into a design matrix.

        Args:
            df (pandas.DataFrame): dataframe with one column per parameter.
            parameters (list[sherpa.core.Parameter]): the parameters.

        Returns:
            numpy.darray: the design matrix.
        """
        X = []
        self.Xcolumns = {}
        column_count = 0
        for p in parameters:
            if isinstance(p, Choice) and len(p.range) == 1:
                raise ValueError("Currently constant parameters are not allowed"
                                 " for Bayesian Optimization.")

            elif isinstance(p, Choice):
                self.transformers[
                    p.name] = BayesianOptimization.ChoiceTransformer(p)
                this_feature = self.transformers[p.name].transform(df[p.name])

            elif isinstance(p, Continuous):
                self.transformers[
                    p.name] = BayesianOptimization.ContinuousTransformer(p)
                this_feature = self.transformers[p.name].transform(
                    df[p.name]).reshape((-1, 1))
            elif isinstance(p, Discrete):
                self.transformers[
                    p.name] = BayesianOptimization.DiscreteTransformer(p)
                this_feature = self.transformers[p.name].transform(
                    df[p.name]).reshape((-1, 1))

            self.Xcolumns[p.name] = list(range(column_count, column_count
                                               + this_feature.shape[1]))
            column_count += this_feature.shape[1]
            X.append(this_feature)
        return numpy.concatenate(X, axis=-1)

    def _from_design(self, X):
        """
        Turns a design matrix back into a dataframe.

        Args:
            X (numpy.darray): Design matrix.

        Returns:
            pandas.DataFrame: Dataframe of hyperparameter values.
        """
        columns = {}
        for pname, pcols in self.Xcolumns.items():
            columns[pname] = self.transformers[pname].reverse(numpy.atleast_2d(X)[:, pcols])
        return pandas.DataFrame.from_dict(columns)

    def _expected_improvement(self, y, y_std):
        with numpy.errstate(divide='ignore'):
            scaling_factor = (-1) ** self.lower_is_better
            z = scaling_factor * (y - self.best_y - self.epsilon)/y_std
            expected_improvement = scaling_factor * (y - self.best_y -
                                                     self.epsilon)*scipy.stats.norm.cdf(z) + y_std*scipy.stats.norm.pdf(z)
        return expected_improvement

    def get_expected_improvement(self, X):
        """
        Args:
            X (numpy.ndarray): the continuous parameters.

        Returns:
            numpy.ndarray: expected improvement for x and args.
        """
        y, y_std = self.gp.predict(numpy.atleast_2d(X), return_std=True)
        return self._expected_improvement(y, y_std)

    def _maximize(self, X, max_function):
        """
        Numerically optimize continuous/discrete columns for each row in X.

        Args:
            X (numpy.ndarray): the rows to be optimized, in design form.
            max_function (callable): function to be maximized, takes as input
                rows of X.

        Returns:
            numpy.ndarray: the best solution for each row.
            numpy.ndarray: the respective function values.
        """

        Xoptimized = numpy.zeros_like(X)
        fun_value = numpy.zeros((len(Xoptimized),))

        def _wrapper(x, *args):
            row = self._add_choice(x, *args)
            fval = max_function(row)
            return -1*fval

        for i, row in enumerate(X):
            x0, args = self._strip_choice(row)
            result = scipy.optimize.minimize(fun=_wrapper,
                                             x0=x0,
                                             method='L-BFGS-B',
                                             args=args,
                                             bounds=[(0, 1)] * x0.shape[0])

            Xoptimized[i] = self._add_choice(result.x, *args)
            fun_value[i] = result.fun
        return Xoptimized, -1*fun_value

    def _strip_choice(self, arow):
        """
        Separate choice variables from continuous and discrete.

        Args:
            row (numpy.ndarray): one row of the design matrix.

        Returns:
            numpy.ndarray: values for continuous/discrete variables
            tuple[numpy.ndarray]: values for choice variables
        """
        x = []
        args = []
        for pname, pcols in self.Xcolumns.items():
            if len(pcols) == 1:
                x.append(arow[pcols])
            else:
                args.append(arow[pcols])
        return numpy.array(x), tuple(args)

    def _add_choice(self, x, *args):
        xq = collections.deque(x)
        argsq = collections.deque(args)
        row = numpy.array([])
        for pname, pcols in self.Xcolumns.items():
            if len(pcols) == 1:
                row = numpy.append(row, xq.popleft())
            else:
                row = numpy.append(row, argsq.popleft())
        return row


class PopulationBasedTraining(Algorithm):
    """
    Population based training (PBT) as introduced by Jaderberg et al. 2017.

    PBT trains a generation of ``population_size`` seed trials (randomly initialized) for a user
    specified number of iterations. After that the same number of trials are
    sampled from the top 33% of the seed generation. Those trials are perturbed
    in their hyperparameter configuration and continue training. After that
    trials are sampled from that generation etc.

    Args:
        population_size (int): the number of randomly intialized trials at the
            beginning and number of concurrent trials after that.
        parameter_range (dict[Union[list,tuple]): upper and lower bounds beyond
            which parameters cannot be perturbed.
        perturbation_factors (tuple[float]): the factors by which continuous
            parameters are multiplied upon perturbation; one is sampled randomly
            at a time.
    """
    def __init__(self, population_size=20, parameter_range={}, perturbation_factors=(0.8, 1.0, 1.2)):
        self.population_size = population_size
        self.parameter_range = parameter_range
        self.perturbation_factors = perturbation_factors
        self.generation = 0
        self.count = 0
        self.random_sampler = RandomSearch()

    def load(self, num_trials):
        self.count = num_trials
        self.generation = self.count//self.population_size + 1

    def get_suggestion(self, parameters, results, lower_is_better):
        self.count += 1

        if self.count % self.population_size == 1:
            self.generation += 1

        if self.generation == 1:
            trial = self.random_sampler.get_suggestion(parameters,
                                                        results, lower_is_better)
            trial['lineage'] = ''
            trial['load_from'] = ''
            trial['save_to'] = str(self.count)  # TODO: unifiy with Trial-ID
        else:
            candidate = self._get_candidate(parameters=parameters,
                                            results=results,
                                            lower_is_better=lower_is_better)
            trial = self._perturb(candidate=candidate, parameters=parameters)
            trial['load_from'] = str(int(trial['save_to']))
            trial['save_to'] = str(int(self.count))
            trial['lineage'] += trial['load_from'] + ','

        return trial

    def _get_candidate(self, parameters, results, lower_is_better):
        """
        Samples candidates from the top 33% of population.

        Returns
            dict: parameter dictionary.
        """
        # Select correct generation
        completed = results.loc[results['Status'] != 'INTERMEDIATE', :]
        fr_ = (self.generation - 2) * self.population_size + 1
        to_ = (self.generation - 1) * self.population_size
        population = completed.loc[(completed['Trial-ID'] >= fr_) & (completed['Trial-ID'] <= to_)]

        # Sample from top 33%
        population = population.sort_values(by='Objective', ascending=lower_is_better)
        idx = numpy.random.randint(low=0, high=self.population_size//3)
        d = population.iloc[idx].to_dict()
        trial = {param.name: d[param.name] for param in parameters}
        for key in ['load_from', 'save_to', 'lineage']:
            trial[key] = d[key]
        return trial

    def _perturb(self, candidate, parameters):
        """
        Randomly perturbs candidate parameters by perturbation factors.

        Args:
            candidate (dict): candidate parameter configuration.
            parameters (list[sherpa.core.Parameter]): parameter ranges.

        Returns:
            dict: perturbed parameter configuration.
        """
        for param in parameters:
            if isinstance(param, Continuous) or isinstance(param, Discrete):
                factor = numpy.random.choice(self.perturbation_factors)
                candidate[param.name] *= factor

                if isinstance(param, Discrete):
                    candidate[param.name] = int(candidate[param.name])

                candidate[param.name] = numpy.clip(candidate[param.name],
                                                   min(self.parameter_range.get(param.name) or param.range),
                                                   max(self.parameter_range.get(param.name) or param.range))

            elif isinstance(param, Ordinal):
                shift = numpy.random.choice([-1, 0, +1])
                values = self.parameter_range.get(param.name) or param.range
                newidx = values.index(candidate[param.name]) + shift
                newidx = numpy.clip(newidx, 0, len(values)-1)
                candidate[param.name] = values[newidx]

            elif isinstance(param, Choice):
                candidate[param.name] = param.sample()

            else:
                raise ValueError("Unrecognized Parameter Object.")

        return candidate
