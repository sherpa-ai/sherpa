import os
import numpy
import logging
import pandas
import scipy.stats
import scipy.optimize
import sklearn.gaussian_process
from .core import Choice, Continuous, Discrete, Ordinal
import sklearn.model_selection


logging.basicConfig(level=logging.DEBUG)
alglogger = logging.getLogger(__name__)


class Algorithm(object):
    """
    Abstract algorithm that returns next parameters conditional on parameter
    ranges and previous results.
    """
    def get_suggestion(self, parameters, results, lower_is_better):
        """
        Returns a suggestion for parameter values based on results.

        # Arguments:
            parameters (list[sherpa.Parameter]): the parameters.
            results (pandas.DataFrame): all results so far.

        # Returns:
            (dict) of parameter values.
        """
        raise NotImplementedError("Algorithm class is not usable itself.")

    def load(self, num_trials):
        """
        Reinstantiates the algorithm when loaded.

        # Arguments:
            num_trials (int): number of trials in study so far.
        """
        pass


class RandomSearch(Algorithm):
    """
    Regular Random Search.

    Expects to set a number of trials to yield.
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


class GridSearch(Algorithm):
    """
    Regular Grid Search. Expects Choice parameters.
    
    Example:
    ```
    hp_space = {'act': ['tanh', 'relu'],
                'lrinit': [0.1, 0.01],
                }
    parameters = sherpa.Parameter.grid(hp_space)
    alg = sherpa.algorithms.GridSearch()
    ```
    """
    def __init__(self):
        self.count = 0
        self.grid = None

    def get_suggestion(self, parameters, results=None, lower_is_better=True):
        assert all(isinstance(p, Choice) for p in parameters), "Only Choice Parameters can be used with GridSearch"
        if self.count == 0:
            param_dict = {p.name: p.range for p in parameters}
            self.grid = list(sklearn.model_selection.ParameterGrid(param_dict))
        if self.count == len(self.grid):
            return None
        else:
            params = self.grid[self.count]
            self.count += 1
            return params


class LocalSearch(Algorithm):
    """
    Local Search by Peter with perturbation modified
    """
    def __init__(self, num_random_seeds=10, seed_configurations=[]):
        # num_random_seeds + len(seed_configurations) needs to be larger than max_concurrent
        self.num_random_seeds = num_random_seeds
        self.seed_configurations = seed_configurations
        self.count = 0
        self.random_sampler = RandomSearch(self.num_random_seeds)

    def get_suggestion(self, parameters, results, lower_is_better):
        self.count += 1
        if self.count <= len(self.seed_configurations) + self.num_random_seeds:
            if len(self.seed_configurations) >= self.count:
                return self.seed_configurations[self.count-1]
            else:
                return self.random_sampler.get_suggestion(parameters, results,
                                                          lower_is_better)

        # Get best result so far
        try:
            best_idx = (results.loc[:, 'Objective'].argmin() if lower_is_better
                        else results.loc[:, 'Objective'].argmax())
        except ValueError:
            return self.random_sampler.get_suggestion(parameters,
                                                      results, lower_is_better)

        parameter_names = [p.name for p in parameters]
        best_params = results.loc[best_idx,
                                  parameter_names].to_dict()
        new_params = best_params
        # randomly choose one of the parameters and perturb it
        # while parameter in existing results
        # choose one dimension randomly and resample it
        alglogger.debug(new_params)
        while results.loc[:, parameter_names].isin({key: [value] for key, value in new_params.items()}).apply(all, axis=1).any():
            new_params = best_params.copy()
            p = numpy.random.choice(list(parameters))
            new_params[p.name] = p.sample()
            alglogger.debug(new_params)

        return new_params


class StoppingRule(object):
    """
    Abstract class to evaluate whether a trial should stop conditional on all
    results so far.
    """
    def should_trial_stop(self, trial, results, lower_is_better):
        """
        # Arguments:
            trial (sherpa.Trial): trial to be stopped.
            results (pandas.DataFrame): all results so far.
            lower_is_better (bool): whether lower objective values are better.

        # Returns:
            (bool) decision.
        """
        raise NotImplementedError("StoppingRule class is not usable itself.")


class MedianStoppingRule(StoppingRule):
    """
    Median Stopping-Rule similar to Golovin et al.
    "Google Vizier: A Service for Black-Box Optimization".
    
    # Description:
    For a Trial `t`, the best objective value is found.

    Then the best objective value for every other trial is found.

    Finally, the best-objective for the trial is compared to
    the median of the best-objectives of all other trials.

    If trial `t`'s best objective is worse than that median, it is
    stopped.

    If `t` has not reached the minimum iterations or there are not
    yet the requested number of comparison trials, `t` is not
    stopped. If `t` is all nan's it is stopped by default.
    """
    def __init__(self, min_iterations=0, min_trials=1):
        self.min_iterations = min_iterations
        self.min_trials = min_trials

    def should_trial_stop(self, trial, results, lower_is_better):
        """
        # Arguments:
            trial (sherpa.Trial): trial to be stopped.
            results (pandas.DataFrame): all results so far.
            lower_is_better (bool): whether lower objective values are better.

        # Returns:
            (bool) decision.
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

    ```
        parameters, results, lower_is_better = sherpa.algorithms.get_sample_results_and_params()
    ```

    to get a sample set of parameters, results and lower_is_better variable.
    Useful for algorithm development.

    Note: losses are obtained from
    ```
        loss = param_a / float(iteration + 1) * param_b
    ```
    """
    here = os.path.abspath(os.path.dirname(__file__))
    results = pandas.read_csv(os.path.join(here, "sample_results.csv"), index_col=0)
    parameters = [Choice(name="param_a",
                         range=[1, 2, 3]),
                  Continuous(name="param_b",
                         range=[0, 1])]
    lower_is_better = True
    return parameters, results, lower_is_better


class GaussianProcessEI(Algorithm):
    def __init__(self, num_random_seeds=10, max_num_trials=None,
                 fine_tune=True):
        self.num_random_seeds = num_random_seeds
        self.count = 0
        self.seed_configurations = []
        self.num_spray_samples = 10000
        self.fine_tune = fine_tune
        self.max_num_trials = max_num_trials
        self.random_sampler = RandomSearch()
        self.xtypes = {}
        self.xnames = {}
        self.best_y = None
        self.epsilon=0.00001
        self.lower_is_better = None
        self.gp = None

    def load(self, num_trials):
        self.count = num_trials

    def get_suggestion(self, parameters, results=None,
                       lower_is_better=True):
        self.count += 1
        if self.max_num_trials and self.max_num_trials == self.count:
            return None

        self.lower_is_better = lower_is_better

        if not self.seed_configurations:
            self.generate_seeds(parameters)

        if self.count <= len(self.seed_configurations):
            return self.seed_configurations[self.count-1]
        
        if len(results) == 0 or len(results.loc[results['Status'] != 'INTERMEDIATE', :]) < 1:
            # Warn user: more workers than seed configurations
            return self.random_sampler.get_suggestion(parameters, results, lower_is_better)

        x, y = self.get_input_output_pairs(results, parameters)
        self.best_y = y.min() if lower_is_better else y.max()
        self.gp = sklearn.gaussian_process.GaussianProcessRegressor(kernel=sklearn.gaussian_process.kernels.Matern(nu=2.5))
        self.gp.fit(X=x, y=y)

        xcand, paramscand = self.generate_candidates(parameters)
        ycand, ycand_std = self.gp.predict(xcand, return_std=True)

        ei = self.get_expected_improvement(ycand, ycand_std)
        # print("Max EI: ", ei.max(), paramscand.iloc[numpy.argmax(ei)].to_dict())
        max_ei_idxs = ei.argsort()[-50:][::-1]  # use top 5

        if self.fine_tune:
            return self.fine_tune_candidates(xcand, paramscand, max_ei_idxs, ei)
        else:
            return paramscand.iloc[numpy.argmax(ei)].to_dict()

    def generate_seeds(self, parameters):
        choice_grid_search = GridSearch()
        choice_params = [p for p in parameters if isinstance(p, Choice)]
        other_params = [p for p in parameters if not isinstance(p, Choice)]
        p = choice_grid_search.get_suggestion(choice_params)
        while p:
            p.update(self.random_sampler.get_suggestion(other_params))
            self.seed_configurations.append(p)
            p = choice_grid_search.get_suggestion(choice_params)

        while len(self.seed_configurations) < self.num_random_seeds:
            p = self.random_sampler.get_suggestion(parameters)
            self.seed_configurations.append(p)

    def get_input_output_pairs(self, results, parameters):
        completed = results.loc[results['Status'] != 'INTERMEDIATE', :]
        x = completed.loc[:, [p.name for p in parameters]]
        x = self.get_design_matrix(x, parameters)
        y = numpy.array(completed.loc[:, 'Objective'])
        return x, y

    def generate_candidates(self, parameters):
        d = {p.name: [] for p in parameters}
        for _ in range(self.num_spray_samples):
            params = self.random_sampler.get_suggestion(parameters)
            for pname in params:
                d[pname].append(params[pname])
        df = pandas.DataFrame.from_dict(d)
        return self.get_design_matrix(df, parameters), df

    def get_design_matrix(self, df, parameters):
        self.xtypes = {}
        self.xscales = {}
        self.xnames = {}
        self.xbounds = []
        num_samples = len(df)
        num_features = sum(len(p.range) if isinstance(p, Choice) else 1 for p in parameters)
        x = numpy.zeros((num_samples, num_features))
        col = 0
        for p in parameters:
            if isinstance(p, Choice) and len(p.range) == 1:
                continue
            if not isinstance(p, Choice):
                x[:, col] = numpy.array(df[p.name])
                if p.scale == 'log':
                    x[:, col] = numpy.log10(x[:, col])
                self.xtypes[col] = 'continuous' if isinstance(p, Continuous) else 'discrete'
                self.xscales[col] = p.scale
                self.xnames[col] = p.name
                if isinstance(p, Continuous):
                    self.xbounds.append((p.range[0], p.range[1]))
                col += 1
            else:
                for i, val in enumerate(p.range):
                    x[:, col] = numpy.array(1.*(df[p.name] == val))
                    self.xtypes[col] = 'discrete'
                    self.xnames[col] = p.name + '_' + str(i)
                    col += 1
        return x[:, :col]

    def get_expected_improvement(self, y, y_std):
        with numpy.errstate(divide='ignore'):
            scaling_factor = (-1) ** self.lower_is_better
            z = scaling_factor * (y - self.best_y - self.epsilon)/y_std
            expected_improvement = scaling_factor * (y - self.best_y -
                                                     self.epsilon)*scipy.stats.norm.cdf(z)
        return expected_improvement

    def fine_tune_candidates(self, xcand, paramscand, cand_idxs, ei):
        """
        Numerically optimizes the top k candidates.

        # Returns:
            (dict) best candidate
        """
        def continuous_cols(x):
            return numpy.array([x[i] for i in range(len(x)) if self.xtypes[i] == 'continuous'])

        def designrow(xstar, row):
            count = 0
            newrow = numpy.copy(row)
            for i in range(len(row)):
                if self.xtypes[i] == 'continuous':
                    newrow[i] = xstar[count]
                    count += 1
            return newrow.reshape(1, -1)

        def eval_neg_ei(x, row):
            """
            # Arguments:
                x (ndarray): the continuous parameters.
                args: the choice and discrete parameters.

            # Returns:
                (float) expected improvement for x and args.
            """
            y, y_std = self.gp.predict(designrow(x, row), return_std=True)
            return -self.get_expected_improvement(y, y_std)

        best_idx = None
        max_ei = -numpy.inf
        for idx in cand_idxs:
            candidate = xcand[idx]

            res = scipy.optimize.minimize(fun=eval_neg_ei,
                                          x0=continuous_cols(candidate),
                                          args=(candidate,),
                                          method="Nelder-Mead")

            if -res.fun > max_ei:
                max_ei = -res.fun
                best_idx = idx
                for col in range(len(candidate)):
                    if self.xtypes[col] == 'continuous' and self.xscales[col] == 'linear':
                        paramscand.set_value(index=idx, col=self.xnames[col], value=candidate[col])
                    elif self.xtypes[col] == 'continuous' and self.xscales[col] == 'log':
                        paramscand.set_value(index=idx, col=self.xnames[col],
                                             value=10**candidate[col])

        return paramscand.iloc[best_idx].to_dict()



class PopulationBasedTraining(Algorithm):
    def __init__(self, population_size=20, parameter_range={}):
        self.population_size = population_size
        self.parameter_range = parameter_range
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
            candidate = self.get_candidate(parameters=parameters,
                                           results=results,
                                           lower_is_better=lower_is_better)
            trial = self.perturb(candidate=candidate, parameters=parameters)
            trial['load_from'] = trial['save_to']
            trial['save_to'] = str(self.count)
            trial['lineage'] += trial['load_from'] + ','

        return trial

    def get_candidate(self, parameters, results, lower_is_better):
        """
        Samples candidates from the top 33% of population.

        # Returns
            (dict) parameter dictionary.
        """
        # Select correct generation
        completed = results.loc[results['Status'] != 'INTERMEDIATE', :]
#         fr_ = (self.generation - 2) * self.population_size + 1
#         to_ = (self.generation - 1) * self.population_size
#         population = completed.loc[(completed['Trial-ID'] >= fr_) & (completed['Trial-ID'] <= to_)]

        # Sample from top 33%
        population = completed.sort_values(by='Objective', ascending=lower_is_better)
        idx = numpy.random.randint(low=0, high=self.population_size//2 + 1)
        d = population.iloc[idx].to_dict()
        trial = {param.name: d[param.name] for param in parameters}
        for key in ['load_from', 'save_to', 'lineage']:
            trial[key] = d[key]
        return trial

    def perturb(self, candidate, parameters):
        for param in parameters:
            if isinstance(param, Continuous) or isinstance(param, Discrete):
                factor = numpy.random.choice([0.8, 1.0, 1.2])

                if param.scale == 'log':
                    candidate[param.name] = 10**(numpy.log10(candidate[param.name]) * factor)
                else:
                    candidate[param.name] *= factor

                if isinstance(param, Discrete):
                    candidate[param.name] = int(candidate[param.name])

                candidate[param.name] = max(
                    [candidate[param.name], min(self.parameter_range.get(param.name) or param.range)])
                candidate[param.name] = min(
                    [candidate[param.name], max(self.parameter_range.get(param.name) or param.range)])

            elif isinstance(param, Ordinal):
                shift = numpy.random.choice([-1, 0, +1])
                values = self.parameter_range.get(param.name) or param.range
                newidx = values.index(candidate[param.name]) + shift
                newidx = min([newidx, len(values)-1])
                newidx = max([newidx, 0])
                candidate[param.name] = values[newidx]

            elif isinstance(param, Choice):
                continue

            else:
                raise ValueError("Unrecognized Parameter Object.")

        return candidate



