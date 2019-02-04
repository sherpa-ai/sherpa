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


logging.basicConfig(level=logging.WARNING)
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
                 max_concurrent=10, verbosity=False, max_num_trials=None):
        self.model_type = model_type
        assert (num_initial_data_points == 'infer'
                or isinstance(num_initial_data_points, int)),\
            "num_initial_data_points needs to be 'infer' or int."
        self.num_initial_data_points = num_initial_data_points
        self._num_initial_data_points = -1
        self.initial_data_points = initial_data_points
        self.acquisition_type = acquisition_type
        self.max_concurrent = max_concurrent
        self.verbosity = verbosity

        self.next_trials = collections.deque()
        self.num_points_seen_by_model = 0

        self.random_search = sherpa.algorithms.RandomSearch()

        self.domain = []
        self.transformations = {}

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

            self.transformations = self._initialize_transforms(parameters)
            self.domain = self._initialize_domain(parameters,
                                                  self.transformations)

        num_completed_trials = (len(results.query("Status == 'COMPLETED'"))
                                if len(results) > 0 else 0)

        if (num_completed_trials > self._num_initial_data_points
           and num_completed_trials > self.num_points_seen_by_model):
            # generate a new batch from bayes opt and set it as next
            # observations
            self.next_trials.clear()
            X, y = self._prepare_data_for_bayes_opt(parameters, results,
                                                    self.transformations)

            batch = self._generate_bayesopt_batch(self.domain, X, y,
                                                  lower_is_better)

            batch_list_of_dicts = self._reverse_to_sherpa_format(batch,
                                                                 self.transformations,
                                                                 parameters)

            self.next_trials.extend(batch_list_of_dicts)
            self.num_points_seen_by_model = num_completed_trials

        if len(self.next_trials) == 0:
            random_trial = self.random_search.get_suggestion(parameters,
                                                             results,
                                                             lower_is_better)
            self.next_trials.append(random_trial)

        return self.next_trials.popleft()

    def _generate_bayesopt_batch(self, domain, X, y, lower_is_better):
        bo_step = gpyopt_package.methods.BayesianOptimization(f=None,
                                                              domain=domain,
                                                              X=X, Y=y,
                                                              acquisition_type=self.acquisition_type,
                                                              model_type=self.model_type,
                                                              evaluator_type='local_penalization',
                                                              batch_size=self.max_concurrent,
                                                              verbosity=self.verbosity,
                                                              maximize=not lower_is_better)

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
    def _prepare_data_for_bayes_opt(parameters, results, transformations):
        completed = results.query("Status == 'COMPLETED'")
        X = numpy.zeros((len(completed), len(parameters)))
        for i, p in enumerate(parameters):
            X[:, i] = completed[p.name].apply(transformations[p.name].transform)

        y = numpy.array(completed.Objective).reshape((-1, 1))
        return X, y

    class Transform(object):
        def __init__(self, log=False, discrete=False):
            self.log = log
            self.discrete = discrete

        def transform(self, x):
            if self.log:
                x = numpy.log10(x)
            return x

        def reverse(self, x):
            if self.log:
                x = 10**x
            if self.discrete:
                x = numpy.round(x).astype('int')
            return x

    class ChoiceTransform(Transform):
        def __init__(self, vals):
            GPyOpt.Transform.__init__(self)
            self.vals = vals

        def transform(self, element):
            return self.vals.index(element)

        def reverse(self, element):
            return self.vals[int(element)]

    @staticmethod
    def _initialize_transforms(parameters):
        transformations = collections.defaultdict(dict)
        for i, p in enumerate(parameters):
            if isinstance(p, Choice) or isinstance(p, Ordinal):
                transformations[p.name] = GPyOpt.ChoiceTransform(vals=p.range)

            else:
                transformations[p.name] = GPyOpt.Transform(log=p.scale == 'log',
                                                           discrete=isinstance(
                                                               p, Discrete))
        return transformations

    @staticmethod
    def _initialize_domain(parameters, transformations):
            domain = []
            for p in parameters:
                if isinstance(p, Choice) or isinstance(p, Ordinal):
                    domain_type = 'discrete'
                else:
                    domain_type = 'continuous'

                domain.append({'name': p.name, 'type': domain_type,
                               'domain': tuple(transformations[p.name].transform(item)
                                               for item in p.range)})
            return domain

    @staticmethod
    def _reverse_to_sherpa_format(X_next, transformations, parameters):
        # turn batch into list of dictionaries
        df = pandas.DataFrame({p.name: list(map(
            transformations[p.name].reverse, list(X_next[:, i])))
            for i, p in enumerate(parameters)})

        return list(df.T.to_dict().values())



    # class HyperparameterOptions(object):
    #
    #     def __init__(self, verbose=1):
    #         self.index_dict = {}
    #         self.search_space = []
    #         self.verbose = verbose
    #
    #     def fit(self, parameters):
    #         for p in parameters:
    #             if isinstance(p, Continuous):
    #                 if p.scale == 'log':
    #                     self.add_param(name=p.name, domain_type='continuous',
    #                                    domain=(numpy.log10(p.range[0]),
    #                                            numpy.log10(p.range[1])))
    #                 else:
    #                     self.add_param(name=p.name)
    #
    #
    #     def add_param(self, name, domain, domain_type='discrete', enable=True,
    #                   required=True, default=None):
    #         """
    #
    #         # Arguments
    #
    #             search_space: list of hyperparameter configurations required by BayseanOptimizer
    #             index_dict: dictionary that will be used to lookup real values
    #                 and types when we get the hyperopt callback with ints or floats
    #             enable: this parameter will be part of hyperparameter search
    #             required: this parameter must be passed to the model
    #             default: default value if required
    #
    #         """
    #         if self.search_space is None:
    #             self.search_space = []
    #         if self.index_dict is None:
    #             self.index_dict = {'current_index': 0}
    #         if 'current_index' not in self.index_dict:
    #             self.index_dict['current_index'] = 0
    #
    #         if enable:
    #             param_index = self.index_dict['current_index']
    #             numerical_domain = domain
    #             needs_reverse_lookup = False
    #             lookup_as = float
    #             # convert string domains to a domain of integer indexes
    #             if domain_type == 'discrete':
    #                 if isinstance(domain, list) and isinstance(domain[0], str):
    #                     numerical_domain = [i for i in range(len(domain))]
    #                     lookup_as = str
    #                     needs_reverse_lookup = True
    #                 elif isinstance(domain, list) and isinstance(domain[0], bool):
    #                     numerical_domain = [i for i in range(len(domain))]
    #                     lookup_as = bool
    #                     needs_reverse_lookup = True
    #                 elif isinstance(domain, list) and isinstance(domain[0], float):
    #                     lookup_as = float
    #                 else:
    #                     lookup_as = int
    #
    #             opt_dict = {
    #                 'name': name,
    #                 'type': domain_type,
    #                 'domain': numerical_domain}
    #
    #             if enable:
    #                 self.search_space += [opt_dict]
    #                 # create a second version for us to construct the real function call
    #                 opt_dict = copy.deepcopy(opt_dict)
    #                 opt_dict['lookup_as'] = lookup_as
    #             else:
    #                 opt_dict['lookup_as'] = None
    #
    #             opt_dict['enable'] = enable
    #             opt_dict['required'] = required
    #             opt_dict['default'] = default
    #             opt_dict['index'] = param_index
    #             opt_dict['domain'] = domain
    #             opt_dict['needs_reverse_lookup'] = needs_reverse_lookup
    #             self.index_dict[name] = opt_dict
    #             self.index_dict['current_index'] += 1
    #
    #     def params_to_args(self, x):
    #         """ Convert GPyOpt Bayesian Optimizer params back into function call arguments
    #
    #         Arguments:
    #
    #             x: the callback parameter of the GPyOpt Bayesian Optimizer
    #             index_dict: a dictionary with all the information necessary to convert back to function call arguments
    #         """
    #         if len(x.shape) == 1:
    #             # if we get a 1d array convert it to 2d so we are consistent
    #             x = numpy.expand_dims(x, axis=0)
    #         # x is a funky 2d numpy array, so we convert it back to normal parameters
    #         kwargs = {}
    #         for key, opt_dict in six.iteritems(self.index_dict):
    #             if key == 'current_index':
    #                 continue
    #
    #             if opt_dict['enable']:
    #                 arg_name = opt_dict['name']
    #                 optimizer_param_column = opt_dict['index']
    #                 if optimizer_param_column > x.shape[-1]:
    #                     raise ValueError('Attempting to access optimizer_param_column' + str(optimizer_param_column) +
    #                                      ' outside parameter bounds' + str(x.shape) +
    #                                      ' of optimizer array with index dict: ' + str(self.index_dict) +
    #                                      'and array x: ' + str(x))
    #                 param_value = x[:, optimizer_param_column]
    #                 if opt_dict['type'] == 'discrete':
    #                     # the value is an integer indexing into the lookup dict
    #                     if opt_dict['needs_reverse_lookup']:
    #                         domain_index = int(param_value)
    #                         domain_value = opt_dict['domain'][domain_index]
    #                         value = opt_dict['lookup_as'](domain_value)
    #                     else:
    #                         value = opt_dict['lookup_as'](param_value)
    #
    #                 else:
    #                     # the value is a param to use directly
    #                     value = opt_dict['lookup_as'](param_value)
    #
    #                 kwargs[arg_name] = value
    #             elif opt_dict['required']:
    #                 kwargs[opt_dict['name']] = opt_dict['default']
    #         return kwargs
    #
    #     def get_domain(self):
    #         """ Get the hyperparameter search space in the gpyopt domain format.
    #         """
    #         return self.search_space