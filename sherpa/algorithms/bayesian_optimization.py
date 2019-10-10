import numpy
import logging
import sherpa
from sherpa.algorithms import Algorithm
import pandas
from sherpa.core import Choice, Continuous, Discrete, Ordinal
import collections
import GPyOpt as gpyopt_package
import GPy
import warnings


bayesoptlogger = logging.getLogger(__name__)


class GPyOpt(Algorithm):
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
        """
        Infers number of initial data points, or overwrites and warns user if
        she defined less than the number of points needed.
        """
        if num_initial_data_points == 'infer':
            return len(parameters) + 1
        elif num_initial_data_points >= len(parameters):
            return num_initial_data_points
        else:
            warnings.warn("num_initial_data_points < number of "
                                "parameters found. Setting "
                                "num_initial_data_points to "
                                "len(parameters)+1.", UserWarning)
            return len(parameters) + 1

    @staticmethod
    def _process_initial_data_points(initial_data_points, parameters):
        """
        Turns initial_data_points into list of dicts (if Pandas.DataFrame) and
        assures that all defined parameters have settings in the
        initial_data_points.
        """
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
            transform = ParameterTransform.from_parameter(p)
            historical_data = completed[p.name]
            X[:, i] = transform.sherpa_format_to_gpyopt_design_format(
                historical_data)

        y = numpy.array(completed.Objective).reshape((-1, 1))
        if 'varObjective' in completed.columns:
            y_var = numpy.array(completed.varObjective).reshape((-1, 1))
        else:
            y_var = None
        return X, y, y_var

    @staticmethod
    def _initialize_domain(parameters):
        """
        Turn Sherpa parameter definitions into GPyOpt parameter definitions.
        """
        domain = []
        for p in parameters:
            domain.append(
                ParameterTransform.from_parameter(p).to_gpyopt_domain())
        return domain

    @staticmethod
    def _reverse_to_sherpa_format(X_next, parameters):
        """
        Turn design matrix from GPyOpt back into a list of dictionaries with
        Sherpa-style parameters.
        """
        col_dict = {}
        for i, p in enumerate(parameters):
            transform = ParameterTransform.from_parameter(p)
            col_dict[p.name] = transform.gpyopt_design_format_to_list_in_sherpa_format(X_next[:, i])

        return list(pandas.DataFrame(col_dict).T.to_dict().values())


class ParameterTransform(object):
    """
    ParamterTransform base class, creates correct object
    depending on parameter.
    """
    def __init__(self, parameter):
        self.parameter = parameter

    @staticmethod
    def from_parameter(parameter):
        if isinstance(parameter, Choice) or isinstance(parameter, Ordinal):
            return ChoiceTransform(parameter)
        elif isinstance(parameter, Continuous):
            if parameter.scale == 'log':
                return LogContinuousTransform(parameter)
            else:
                return ContinuousTransform(parameter)
        elif isinstance(parameter, Discrete):
            if parameter.scale == 'log':
                warnings.warn("GPyOpt discrete parameter does not "
                              "support log-scale.", UserWarning)
            return DiscreteTransform(parameter)

    def to_gpyopt_domain(self):
        raise NotImplementedError

    def gpyopt_design_format_to_list_in_sherpa_format(self, x):
        raise NotImplementedError

    def sherpa_format_to_gpyopt_design_format(self, x):
        raise NotImplementedError


class ContinuousTransform(ParameterTransform):
    """
    Transforms/reverses Continuous variables.
    """
    def to_gpyopt_domain(self):
        return {'name': self.parameter.name,
                'type': 'continuous',
                'domain': tuple(self.parameter.range)}

    def gpyopt_design_format_to_list_in_sherpa_format(self, x):
        return x

    def sherpa_format_to_gpyopt_design_format(self, x):
        return x


class LogContinuousTransform(ParameterTransform):
    """
    Transforms/reverses Continuous variables if on log-scale.
    """
    def to_gpyopt_domain(self):
        return {'name': self.parameter.name,
                'type': 'continuous',
                'domain': (numpy.log10(self.parameter.range[0]),
                           numpy.log10(self.parameter.range[1]))}

    def gpyopt_design_format_to_list_in_sherpa_format(self, x):
        return 10**x

    def sherpa_format_to_gpyopt_design_format(self, x):
        return numpy.log10(x)


class ChoiceTransform(ParameterTransform):
    """
    Transforms/reverses Choice variables to numeric choices since GPyOpt
    does not accept string choices.
    """
    def to_gpyopt_domain(self):
        return {'name': self.parameter.name, 'type': 'categorical',
                'domain': numpy.array(range(len(self.parameter.range)))}

    def gpyopt_design_format_to_list_in_sherpa_format(self, x):
        return [self.parameter.range[int(elem)] for elem in x]

    def sherpa_format_to_gpyopt_design_format(self, x):
        return [self.parameter.range.index(elem) for elem in x]


class DiscreteTransform(ParameterTransform):
    """
    Transforms Discrete parameter from/to GPyOpt
    """
    def to_gpyopt_domain(self):
        return {'name': self.parameter.name,
                'type': 'discrete',
                'domain': tuple(range(self.parameter.range[0],
                                      self.parameter.range[1]+1))}

    def gpyopt_design_format_to_list_in_sherpa_format(self, x):
        return list(x.astype('int'))

    def sherpa_format_to_gpyopt_design_format(self, x):
        return x