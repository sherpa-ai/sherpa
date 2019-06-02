import numpy
import logging
import sherpa
import pandas
import scipy.stats
import scipy.optimize
import sklearn.gaussian_process
from sherpa.core import Choice, Continuous, Discrete, Ordinal
import sklearn.model_selection
from sklearn import preprocessing
import collections
import copy
import six
import GPyOpt as gpyopt_package
import GPy


bayesoptlogger = logging.getLogger(__name__)


class BayesianOptimization(sherpa.algorithms.Algorithm):
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
        self.random_sampler = sherpa.algorithms.RandomSearch()
        self.grid_search = sherpa.algorithms.GridSearch(num_grid_points=num_grid_points)
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

        if self.max_num_trials and self.count > self.max_num_trials:
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


class GPyOpt(sherpa.algorithms.Algorithm):
    """
    Sherpa wrapper around the GPyOpt package
    (https://github.com/SheffieldML/GPyOpt).

    Args:
        model_type (str): The model used:
            - 'GP', standard Gaussian process.
            - 'GP_MCMC', Gaussian process with prior in the hyper-parameters.
            - 'sparseGP', sparse Gaussian process.
            - 'warperdGP', warped Gaussian process.
            - 'InputWarpedGP', input warped Gaussian process
            - 'RF', random forest (scikit-learn).
        num_initial_data_points (int): Number of data points to collect before
            fitting model. Needs to be greater/equal to the number of hyper-
            parameters that are being optimized. Using default 'infer' corres-
            ponds to number of hyperparameters + 1.
        initial_data_points (list[dict] or pandas.Dataframe): Specifies initial
            data points. If len(initial_data_points)<num_initial_data_points
            then the rest is randomly sampled. Use this option to provide
            hyperparameter configurations that are known to be good.
        acquisition_type (str): Type of acquisition function to use.
            - 'EI', expected improvement.
            - 'EI_MCMC', integrated expected improvement (requires GP_MCMC model).
            - 'MPI', maximum probability of improvement.
            - 'MPI_MCMC', maximum probability of improvement (requires GP_MCMC model).
            - 'LCB', GP-Lower confidence bound.
            - 'LCB_MCMC', integrated GP-Lower confidence bound (requires GP_MCMC model).
        max_concurrent (int): The number of concurrent trials. This generates
            a batch of max_concurrent trials from GPyOpt to evaluate. If a new
            observation becomes available, the model is re-evaluated and a new
            batch is created regardless of whether the previous batch was used
            up. The used method is local penalization.
        verbosity (bool): Print models and other options during the optimization.
        max_num_trials (int): maximum number of trials to run for.
    """
    def __init__(self, model_type='GP', num_initial_data_points='infer',
                 initial_data_points=[], acquisition_type='EI',
                 max_concurrent=4, verbosity=False, max_num_trials=None):
        self.model_type = model_type
        assert (num_initial_data_points == 'infer'
                or isinstance(num_initial_data_points, int)),\
            "num_initial_data_points needs to be 'infer' or int."
        self.num_initial_data_points = num_initial_data_points
        self._num_initial_data_points = -1
        self.initial_data_points = initial_data_points
        self.acquisition_type = acquisition_type

        assert model_type != 'GP_MCMC' and acquisition_type != 'EI_MCMC'\
            if max_concurrent > 1 else True,\
            "GPyOpt has a bug for _MCMC with batch size > 1."
        self.max_concurrent = max_concurrent
        self.verbosity = verbosity

        self.next_trials = collections.deque()
        self.num_points_seen_by_model = 0

        self.random_search = sherpa.algorithms.RandomSearch()

        self.domain = []

        self.max_num_trials = max_num_trials
        self.count = 0

    def get_suggestion(self, parameters, results, lower_is_better):
        self.count += 1
        if self.max_num_trials and self.count > self.max_num_trials:
            return None

        # setup
        if self._num_initial_data_points == -1:
            self._num_initial_data_points = self._infer_num_initial_data_points(
                self.num_initial_data_points,
                parameters)

            self.next_trials.extend(
                self._process_initial_data_points(self.initial_data_points,
                                                  parameters))
            self.domain = self._initialize_domain(parameters)

        num_completed_trials = (len(results.query("Status == 'COMPLETED'"))
                                if results is not None and len(results) > 0 else 0)

        if (num_completed_trials >= self._num_initial_data_points
           and num_completed_trials > self.num_points_seen_by_model):
            # generate a new batch from bayes opt and set it as next
            # observations

            # clear previous batch since new data is available
            self.next_trials.clear()

            X, y, y_var = self._prepare_data_for_bayes_opt(parameters, results)

            batch = self._generate_bayesopt_batch(self.domain, X, y, y_var,
                                                  lower_is_better)

            batch_list_of_dicts = self._reverse_to_sherpa_format(batch,
                                                                 parameters)

            self.next_trials.extend(batch_list_of_dicts)
            self.num_points_seen_by_model = num_completed_trials

        if len(self.next_trials) == 0:
            random_trial = self.random_search.get_suggestion(parameters,
                                                             results,
                                                             lower_is_better)
            self.next_trials.append(random_trial)

        return self.next_trials.popleft()

    def _generate_bayesopt_batch(self, domain, X, y, y_var, lower_is_better):
        y_adjusted = y * (-1)**(not lower_is_better)
        if y_var is not None:
            kern = GPy.kern.Matern52(input_dim=X.shape[1], variance=1.) + GPy.kern.Bias(
                X.shape[1])
            m = GPy.models.GPHeteroscedasticRegression(X, y_adjusted, kern)
            m['.*het_Gauss.variance'] = y_var
            m.het_Gauss.variance.fix()
            m.optimize()
            kwargs = {'model': m}
        else:
            kwargs = {'model_type': self.model_type}

        bo_step = gpyopt_package.methods.BayesianOptimization(f=None,
                                                              domain=domain,
                                                              X=X, Y=y_adjusted,
                                                              acquisition_type=self.acquisition_type,
                                                              evaluator_type='local_penalization',
                                                              batch_size=self.max_concurrent,
                                                              verbosity=self.verbosity,
                                                              maximize=False,
                                                              exact_feval=False,
                                                              **kwargs)
        return bo_step.suggest_next_locations()

    @staticmethod
    def _infer_num_initial_data_points(num_initial_data_points, parameters):
        if num_initial_data_points == 'infer':
            return len(parameters) + 1
        elif num_initial_data_points >= len(parameters):
            return num_initial_data_points
        else:
            bayesoptlogger.warn("num_initial_data_points < number of "
                                "parameters found. Setting "
                                "num_initial_data_points to "
                                "len(parameters)+1.")
            return len(parameters) + 1

    @staticmethod
    def _process_initial_data_points(initial_data_points, parameters):
        if isinstance(initial_data_points, pandas.DataFrame):
            _initial_data_points = list(initial_data_points.T.to_dict().values())
        else:
            _initial_data_points = initial_data_points

        for p in parameters:
            if not all(p.name in data_point
                       for data_point in _initial_data_points):
                raise ValueError("Missing parameter in initial_data_point. Check that you " \
                "included all specified hyperparameters.")

        return _initial_data_points

    @staticmethod
    def _prepare_data_for_bayes_opt(parameters, results):
        """
        Turn historical data from Sherpa results dataframe into design matrix
        X and objective values y to be consumed by GPyOpt.
        """
        completed = results.query("Status == 'COMPLETED'")
        X = numpy.zeros((len(completed), len(parameters)))
        for i, p in enumerate(parameters):
            if isinstance(p, Choice) or isinstance(p, Ordinal):
                X[:, i] = completed[p.name].apply(
                    lambda elem: GPyOpt.ChoiceTransform.transform(
                        elem, p.range))
            elif isinstance(p, Continuous) and p.scale == 'log':
                X[:, i] = GPyOpt.LogTransform.transform(completed[p.name])
            else:
                X[:, i] = completed[p.name]

        y = numpy.array(completed.Objective).reshape((-1, 1))
        if 'varObjective' in completed.columns:
            y_var = numpy.array(completed.varObjective).reshape((-1, 1))
        else:
            y_var = None
        return X, y, y_var

    class LogTransform(object):
        """
        Transforms/reverses Continuous variables if on log-scale.
        """
        @staticmethod
        def transform(x):
            return numpy.log10(x)

        @staticmethod
        def reverse(x):
            return 10**x

    class ChoiceTransform(object):
        """
        Transforms/reverses Choice variables to numeric choices since GPyOpt
        does not accept string choices.
        """
        @staticmethod
        def transform(element, vals):
            return vals.index(element)

        @staticmethod
        def reverse(element, vals):
            return vals[int(element)]

    @staticmethod
    def _initialize_domain(parameters):
        """
        Turn Sherpa parameter definitions into GPyOpt parameter definitions.
        """
        domain = []
        for p in parameters:
            if isinstance(p, Choice) or isinstance(p, Ordinal):
                domain.append({'name': p.name, 'type': 'categorical',
                               'domain': numpy.array(
                                   [GPyOpt.ChoiceTransform.transform(
                                       item, p.range)
                                   for item in p.range])})
            elif isinstance(p, Discrete):
                domain.append({'name': p.name, 'type': 'discrete',
                               'domain': tuple(range(p.range[0],
                                                     p.range[1]+1))})
            else:
                if p.scale == 'log':
                    lower_bound = GPyOpt.LogTransform.transform(p.range[0])
                    upper_bound = GPyOpt.LogTransform.transform(p.range[1])
                else:
                    lower_bound = p.range[0]
                    upper_bound = p.range[1]
                domain.append({'name': p.name, 'type': 'continuous',
                               'domain': (lower_bound, upper_bound)})
        return domain

    @staticmethod
    def _reverse_to_sherpa_format(X_next, parameters):
        """
        Turn design matrix from GPyOpt back into a list of dictionaries with
        Sherpa-style parameters.
        """
        col_dict = {}
        for i, p in enumerate(parameters):
            if isinstance(p, Choice) or isinstance(p, Ordinal):
                col_dict[p.name] = list(map(
                    lambda x: GPyOpt.ChoiceTransform.reverse(
                        x, p.range), X_next[:, i]))
            elif isinstance(p, Continuous) and p.scale == 'log':
                col_dict[p.name] = list(GPyOpt.LogTransform.reverse(X_next[:, i]))
            else:
                col_dict[p.name] = list(X_next[:, i])

        return list(pandas.DataFrame(col_dict).T.to_dict().values())