"""GaussianUnivariate module."""
import sys
import pickle
import contextlib
from abc import ABC
import numpy as np
from enum import Enum
# from scipy.stats import kstest
# from scipy.stats import norm
from scipy import stats
from copy import deepcopy
from functools import wraps
import importlib
import pandas as pd
import logging
# from copulas.univariate.base import BoundedType, ParametricType, ScipyModel

class ParametricType(Enum):
    """Parametric Enum."""

    NON_PARAMETRIC = 0
    PARAMETRIC = 1


class BoundedType(Enum):
    """Bounded Enum."""

    UNBOUNDED = 0
    SEMI_BOUNDED = 1
    BOUNDED = 2


def store_args(__init__):
    """Save ``*args`` and ``**kwargs`` used in the ``__init__`` of a copula.

    Args:
        __init__(callable): ``__init__`` function to store their arguments.

    Returns:
        callable: Decorated ``__init__`` function.
    """

    @wraps(__init__)
    def new__init__(self, *args, **kwargs):
        args_copy = deepcopy(args)
        kwargs_copy = deepcopy(kwargs)
        __init__(self, *args, **kwargs)
        self.__args__ = args_copy
        self.__kwargs__ = kwargs_copy

    return new__init__


def validate_random_state(random_state):
    """Validate random state argument.

    Args:
        random_state (int, numpy.random.RandomState, tuple, or None):
            Seed or RandomState for the random generator.

    Output:
        numpy.random.RandomState
    """
    if random_state is None:
        return None

    if isinstance(random_state, int):
        return np.random.RandomState(seed=random_state)
    elif isinstance(random_state, np.random.RandomState):
        return random_state
    else:
        raise TypeError(
            f'`random_state` {random_state} expected to be an int '
            'or `np.random.RandomState` object.'
        )


def get_qualified_name(_object):
    """Return the Fully Qualified Name from an instance or class."""
    module = _object.__module__
    if hasattr(_object, '__name__'):
        _class = _object.__name__
    else:
        _class = _object.__class__.__name__

    return module + '.' + _class


def get_instance(obj, **kwargs):
    """Create new instance of the ``obj`` argument.

    Args:
        obj (str, type, instance):
    """
    instance = None
    if isinstance(obj, str):
        package, name = obj.rsplit('.', 1)
        instance = getattr(importlib.import_module(package), name)(**kwargs)
    elif isinstance(obj, type):
        instance = obj(**kwargs)
    else:
        if kwargs:
            instance = obj.__class__(**kwargs)
        else:
            args = getattr(obj, '__args__', ())
            kwargs = getattr(obj, '__kwargs__', {})
            instance = obj.__class__(*args, **kwargs)

    return instance

@contextlib.contextmanager
def set_random_state(random_state, set_model_random_state):
    """Context manager for managing the random state.

    Args:
        random_state (int or np.random.RandomState):
            The random seed or RandomState.
        set_model_random_state (function):
            Function to set the random state on the model.
    """
    original_state = np.random.get_state()
    np.random.set_state(random_state.get_state())

    try:
        yield
    finally:
        current_random_state = np.random.RandomState()
        current_random_state.set_state(np.random.get_state())
        set_model_random_state(current_random_state)
        np.random.set_state(original_state)

def random_state(function):
    """Set the random state before calling the function.

    Args:
        function (Callable):
            The function to wrap around.
    """

    @wraps(function)
    def wrapper(self, *args, **kwargs):
        if self.random_state is None:
            return function(self, *args, **kwargs)
        else:
            with set_random_state(self.random_state, self.set_random_state):
                return function(self, *args, **kwargs)

    return wrapper



def select_univariate(X, candidates):
    best_ks = np.inf
    best_model = None
    for model in candidates:
        try:
            instance = get_instance(model)
            instance.fit(X)
            ks, _ = stats.kstest(X, instance.cdf)
            if ks < best_ks:
                best_ks = ks
                best_model = model
        except Exception:
            # Distribution not supported
            pass

    return get_instance(best_model)

def check_valid_values(function):
    """Raise an exception if the given values are not supported.

    Args:
        function(callable): Method whose unique argument is a numpy.array-like object.

    Returns:
        callable: Decorated function

    Raises:
        ValueError: If there are missing or invalid values or if the dataset is empty.
    """

    @wraps(function)
    def decorated(self, X, *args, **kwargs):
        if isinstance(X, pd.DataFrame):
            W = X.to_numpy()
        else:
            W = X

        if not len(W):
            raise ValueError('Your dataset is empty.')

        if not (np.issubdtype(W.dtype, np.floating) or np.issubdtype(W.dtype, np.integer)):
            raise ValueError('There are non-numerical values in your data.')

        if np.isnan(W).any().any():
            raise ValueError('There are nan values in your data.')

        return function(self, X, *args, **kwargs)

    return decorated


class Univariate(object):
    """Univariate Distribution.

    Args:
        candidates (list[str or type or Univariate]):
            List of candidates to select the best univariate from.
            It can be a list of strings representing Univariate FQNs,
            or a list of Univariate subclasses or a list of instances.
        parametric (ParametricType):
            If not ``None``, only select subclasses of this type.
            Ignored if ``candidates`` is passed.
        bounded (BoundedType):
            If not ``None``, only select subclasses of this type.
            Ignored if ``candidates`` is passed.
        random_state (int or np.random.RandomState):
            Random seed or RandomState to use.
        selection_sample_size (int):
            Size of the subsample to use for candidate selection.
            If ``None``, all the data is used.
    """

    PARAMETRIC = ParametricType.NON_PARAMETRIC
    BOUNDED = BoundedType.UNBOUNDED

    fitted = False
    _constant_value = None
    _instance = None

    @classmethod
    def _select_candidates(cls, parametric=None, bounded=None):
        """Select which subclasses fulfill the specified constriants.

        Args:
            parametric (ParametricType):
                If not ``None``, only select subclasses of this type.
            bounded (BoundedType):
                If not ``None``, only select subclasses of this type.

        Returns:
            list:
                Selected subclasses.
        """
        candidates = []
        for subclass in cls.__subclasses__():
            candidates.extend(subclass._select_candidates(parametric, bounded))
            if ABC in subclass.__bases__:
                continue
            if parametric is not None and subclass.PARAMETRIC != parametric:
                continue
            if bounded is not None and subclass.BOUNDED != bounded:
                continue

            candidates.append(subclass)

        return candidates

    @store_args
    def __init__(
        self,
        candidates=None,
        parametric=None,
        bounded=None,
        random_state=None,
        selection_sample_size=None,
    ):
        self.candidates = candidates or self._select_candidates(parametric, bounded)
        self.random_state = validate_random_state(random_state)
        self.selection_sample_size = selection_sample_size

    @classmethod
    def __repr__(cls):
        """Return class name."""
        return cls.__name__

    def check_fit(self):
        """Check whether this model has already been fit to a random variable.

        Raise a ``NotFittedError`` if it has not.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        if not self.fitted:
            raise ValueError('This model is not fitted.')

    def _constant_sample(self, num_samples):
        """Sample values for a constant distribution.

        Args:
            num_samples (int):
                Number of rows to sample

        Returns:
            numpy.ndarray:
                Sampled values. Array of shape (num_samples,).
        """
        return np.full(num_samples, self._constant_value)

    def _constant_cumulative_distribution(self, X):
        """Cumulative distribution for the degenerate case of constant distribution.

        Note that the output of this method will be an array whose unique values are 0 and 1.
        More information can be found here: https://en.wikipedia.org/wiki/Degenerate_distribution

        Arguments:
            X (numpy.ndarray):
                Values for which the cumulative distribution will be computed.
                It must have shape (n, 1).

        Returns:
            numpy.ndarray:
                Cumulative distribution values for points in X.
        """
        result = np.ones(X.shape)
        result[np.nonzero(X < self._constant_value)] = 0

        return result

    def _constant_probability_density(self, X):
        """Probability density for the degenerate case of constant distribution.

        Note that the output of this method will be an array whose unique values are 0 and 1.
        More information can be found here: https://en.wikipedia.org/wiki/Degenerate_distribution

        Arguments:
            X (numpy.ndarray):
                Values for which the probability density will be computed.
                It must have shape (n, 1).

        Returns:
            numpy.ndarray:
                Probability density values for points in X.
        """
        result = np.zeros(X.shape)
        result[np.nonzero(X == self._constant_value)] = 1

        return result

    def _constant_percent_point(self, X):
        """Percent point for the degenerate case of constant distribution.

        Note that the output of this method will be an array whose unique values are `np.nan`
        and self._constant_value.
        More information can be found here: https://en.wikipedia.org/wiki/Degenerate_distribution

        Arguments:
            U (numpy.ndarray):
                Values for which the cumulative distribution will be computed.
                It must have shape (n, 1) and values must be in [0,1].

        Returns:
            numpy.ndarray:
                Inverse cumulative distribution values for points in U.
        """
        return np.full(X.shape, self._constant_value)

    def _replace_constant_methods(self):
        """Replace conventional distribution methods by its constant counterparts."""
        self.cumulative_distribution = self._constant_cumulative_distribution
        self.percent_point = self._constant_percent_point
        self.probability_density = self._constant_probability_density
        self.sample = self._constant_sample

    def _set_constant_value(self, constant_value):
        """Set the distribution up to behave as a degenerate distribution.

        The constant value is stored as ``self._constant_value`` and all
        the methods are replaced by their degenerate counterparts.

        Args:
            constant_value (float):
                Value to set as the constant one.
        """
        self._constant_value = constant_value
        self._replace_constant_methods()

    def _check_constant_value(self, X):
        """Check if a Series or array contains only one unique value.

        If it contains only one value, set the instance up to behave accordingly.

        Args:
            X (numpy.ndarray):
                Data to analyze.

        Returns:
            float:
                Whether the input data had only one value or not.
        """
        uniques = np.unique(X)
        if len(uniques) == 1:
            self._set_constant_value(uniques[0])

            return True

        return False
    

    def fit(self, X):
        """Fit the model to a random variable.

        Arguments:
            X (numpy.ndarray):
                Values of the random variable. It must have shape (n, 1).
        """
        if self.selection_sample_size and self.selection_sample_size < len(X):
            selection_sample = np.random.choice(X, size=self.selection_sample_size)
        else:
            selection_sample = X

        self._instance = select_univariate(selection_sample, self.candidates)
        self._instance.fit(X)

        self.fitted = True

    def probability_density(self, X):
        """Compute the probability density for each point in X.

        Arguments:
            X (numpy.ndarray):
                Values for which the probability density will be computed.
                It must have shape (n, 1).

        Returns:
            numpy.ndarray:
                Probability density values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        return self._instance.probability_density(X)

    def log_probability_density(self, X):
        """Compute the log of the probability density for each point in X.

        It should be overridden with numerically stable variants whenever possible.

        Arguments:
            X (numpy.ndarray):
                Values for which the log probability density will be computed.
                It must have shape (n, 1).

        Returns:
            numpy.ndarray:
                Log probability density values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        if self._instance:
            return self._instance.log_probability_density(X)

        return np.log(self.probability_density(X))

    def pdf(self, X):
        """Compute the probability density for each point in X.

        Arguments:
            X (numpy.ndarray):
                Values for which the probability density will be computed.
                It must have shape (n, 1).

        Returns:
            numpy.ndarray:
                Probability density values for points in X.
        """
        return self.probability_density(X)

    def cumulative_distribution(self, X):
        """Compute the cumulative distribution value for each point in X.

        Arguments:
            X (numpy.ndarray):
                Values for which the cumulative distribution will be computed.
                It must have shape (n, 1).

        Returns:
            numpy.ndarray:
                Cumulative distribution values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        return self._instance.cumulative_distribution(X)

    def cdf(self, X):
        """Compute the cumulative distribution value for each point in X.

        Arguments:
            X (numpy.ndarray):
                Values for which the cumulative distribution will be computed.
                It must have shape (n, 1).

        Returns:
            numpy.ndarray:
                Cumulative distribution values for points in X.
        """
        return self.cumulative_distribution(X)

    def percent_point(self, U):
        """Compute the inverse cumulative distribution value for each point in U.

        Arguments:
            U (numpy.ndarray):
                Values for which the cumulative distribution will be computed.
                It must have shape (n, 1) and values must be in [0,1].

        Returns:
            numpy.ndarray:
                Inverse cumulative distribution values for points in U.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        return self._instance.percent_point(U)

    def ppf(self, U):
        """Compute the inverse cumulative distribution value for each point in U.

        Arguments:
            U (numpy.ndarray):
                Values for which the cumulative distribution will be computed.
                It must have shape (n, 1) and values must be in [0,1].

        Returns:
            numpy.ndarray:
                Inverse cumulative distribution values for points in U.
        """
        return self.percent_point(U)

    def set_random_state(self, random_state):
        """Set the random state.

        Args:
            random_state (int, np.random.RandomState, or None):
                Seed or RandomState for the random generator.
        """
        self.random_state = validate_random_state(random_state)

    def sample(self, n_samples=1):
        """Sample values from this model.

        Argument:
            n_samples (int):
                Number of values to sample

        Returns:
            numpy.ndarray:
                Array of shape (n_samples, 1) with values randomly
                sampled from this model distribution.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        return self._instance.sample(n_samples)

    def _get_params(self):
        """Return attributes from self.model to serialize.

        Returns:
            dict:
                Parameters of the underlying distribution.
        """
        return self._instance._get_params()

    def _set_params(self, params):
        """Set the parameters of this univariate.

        Must be implemented in all the subclasses.

        Args:
            dict:
                Parameters to recreate this instance.
        """
        raise NotImplementedError()

    def to_dict(self):
        """Return the parameters of this model in a dict.

        Returns:
            dict:
                Dictionary containing the distribution type and all
                the parameters that define the distribution.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()

        params = self._get_params()
        if self.__class__ is Univariate:
            params['type'] = get_qualified_name(self._instance)
        else:
            params['type'] = get_qualified_name(self)

        return params

    @classmethod
    def from_dict(cls, params):
        """Build a distribution from its params dict.

        Args:
            params (dict):
                Dictionary containing the FQN of the distribution and the
                necessary parameters to rebuild it.
                The input format is exactly the same that is outputted by
                the distribution class ``to_dict`` method.

        Returns:
            Univariate:
                Distribution instance.
        """
        params = params.copy()
        distribution = get_instance(params.pop('type'))
        distribution._set_params(params)
        distribution.fitted = True

        return distribution

    def save(self, path):
        """Serialize this univariate instance using pickle.

        Args:
            path (str):
                Path to where this distribution will be serialized.
        """
        with open(path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    @classmethod
    def load(cls, path):
        """Load a Univariate instance from a pickle file.

        Args:
            path (str):
                Path to the pickle file where the distribution has been serialized.

        Returns:
            Univariate:
                Loaded instance.
        """
        with open(path, 'rb') as pickle_file:
            return pickle.load(pickle_file)



class Multivariate(object):
    """Abstract class for a multi-variate copula object."""

    fitted = False

    def __init__(self, random_state=None):
        self.random_state = validate_random_state(random_state)

    def fit(self, X):
        """Fit the model to table with values from multiple random variables.

        Arguments:
            X (pandas.DataFrame):
                Values of the random variables.
        """
        raise NotImplementedError

    def probability_density(self, X):
        """Compute the probability density for each point in X.

        Arguments:
            X (pandas.DataFrame):
                Values for which the probability density will be computed.

        Returns:
            numpy.ndarray:
                Probability density values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        raise NotImplementedError

    def log_probability_density(self, X):
        """Compute the log of the probability density for each point in X.

        Arguments:
            X (pandas.DataFrame):
                Values for which the log probability density will be computed.

        Returns:
            numpy.ndarray:
                Log probability density values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        return np.log(self.probability_density(X))

    def pdf(self, X):
        """Compute the probability density for each point in X.

        Arguments:
            X (pandas.DataFrame):
                Values for which the probability density will be computed.

        Returns:
            numpy.ndarray:
                Probability density values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        return self.probability_density(X)

    def cumulative_distribution(self, X):
        """Compute the cumulative distribution value for each point in X.

        Arguments:
            X (pandas.DataFrame):
                Values for which the cumulative distribution will be computed.

        Returns:
            numpy.ndarray:
                Cumulative distribution values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        raise NotImplementedError

    def cdf(self, X):
        """Compute the cumulative distribution value for each point in X.

        Arguments:
            X (pandas.DataFrame):
                Values for which the cumulative distribution will be computed.

        Returns:
            numpy.ndarray:
                Cumulative distribution values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        return self.cumulative_distribution(X)

    def set_random_state(self, random_state):
        """Set the random state.

        Args:
            random_state (int, np.random.RandomState, or None):
                Seed or RandomState for the random generator.
        """
        self.random_state = validate_random_state(random_state)

    def sample(self, num_rows=1):
        """Sample values from this model.

        Argument:
            num_rows (int):
                Number of rows to sample.

        Returns:
            numpy.ndarray:
                Array of shape (n_samples, *) with values randomly
                sampled from this model distribution.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        raise NotImplementedError

    def to_dict(self):
        """Return a `dict` with the parameters to replicate this object.

        Returns:
            dict:
                Parameters of this distribution.
        """
        raise NotImplementedError

    @classmethod
    def from_dict(cls, params):
        """Create a new instance from a parameters dictionary.

        Args:
            params (dict):
                Parameters of the distribution, in the same format as the one
                returned by the ``to_dict`` method.

        Returns:
            Multivariate:
                Instance of the distribution defined on the parameters.
        """
        multivariate_class = get_instance(params['type'])
        return multivariate_class.from_dict(params)

    @classmethod
    def load(cls, path):
        """Load a Multivariate instance from a pickle file.

        Args:
            path (str):
                Path to the pickle file where the distribution has been serialized.

        Returns:
            Multivariate:
                Loaded instance.
        """
        with open(path, 'rb') as pickle_file:
            return pickle.load(pickle_file)

    def save(self, path):
        """Serialize this multivariate instance using pickle.

        Args:
            path (str):
                Path to where this distribution will be serialized.
        """
        with open(path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    def check_fit(self):
        """Check whether this model has already been fit to a random variable.

        Raise a ``NotFittedError`` if it has not.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        if not self.fitted:
            raise ValueError('This model is not fitted.')
        

class ScipyModel(Univariate, ABC):
    """Wrapper for scipy models.

    This class makes the probability_density, cumulative_distribution,
    percent_point and sample point at the underlying pdf, cdf, ppd and rvs
    methods respectively.

    fit, _get_params and _set_params must be implemented by the subclasses.
    """

    MODEL_CLASS = None

    _params = None

    def __init__(self, random_state=None):
        """Initialize Scipy model.

        Overwrite Univariate __init__ to skip candidate initialization.

        Args:
            random_state (int, np.random.RandomState, or None): seed
                or RandomState for random generator.
        """
        self.random_state = validate_random_state(random_state)

    def probability_density(self, X):
        """Compute the probability density for each point in X.

        Arguments:
            X (numpy.ndarray):
                Values for which the probability density will be computed.
                It must have shape (n, 1).

        Returns:
            numpy.ndarray:
                Probability density values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        return self.MODEL_CLASS.pdf(X, **self._params)

    def log_probability_density(self, X):
        """Compute the log of the probability density for each point in X.

        Arguments:
            X (numpy.ndarray):
                Values for which the log probability density will be computed.
                It must have shape (n, 1).

        Returns:
            numpy.ndarray:
                Log probability density values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        if hasattr(self.MODEL_CLASS, 'logpdf'):
            return self.MODEL_CLASS.logpdf(X, **self._params)

        return np.log(self.probability_density(X))

    def cumulative_distribution(self, X):
        """Compute the cumulative distribution value for each point in X.

        Arguments:
            X (numpy.ndarray):
                Values for which the cumulative distribution will be computed.
                It must have shape (n, 1).

        Returns:
            numpy.ndarray:
                Cumulative distribution values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        return self.MODEL_CLASS.cdf(X, **self._params)

    def percent_point(self, U):
        """Compute the inverse cumulative distribution value for each point in U.

        Arguments:
            U (numpy.ndarray):
                Values for which the cumulative distribution will be computed.
                It must have shape (n, 1) and values must be in [0,1].

        Returns:
            numpy.ndarray:
                Inverse cumulative distribution values for points in U.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        return self.MODEL_CLASS.ppf(U, **self._params)

    @random_state
    def sample(self, n_samples=1):
        """Sample values from this model.

        Argument:
            n_samples (int):
                Number of values to sample

        Returns:
            numpy.ndarray:
                Array of shape (n_samples, 1) with values randomly
                sampled from this model distribution.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        return self.MODEL_CLASS.rvs(size=n_samples, **self._params)

    def _fit(self, X):
        """Fit the model to a non-constant random variable.

        Must be implemented in all the subclasses.

        Arguments:
            X (numpy.ndarray):
                Values of the random variable. It must have shape (n, 1).
        """
        raise NotImplementedError()

    def fit(self, X):
        """Fit the model to a random variable.

        Arguments:
            X (numpy.ndarray):
                Values of the random variable. It must have shape (n, 1).
        """
        if self._check_constant_value(X):
            self._fit_constant(X)
        else:
            self._fit(X)

        self.fitted = True

    def _get_params(self):
        """Return attributes from self._model to serialize.

        Must be implemented in all the subclasses.

        Returns:
            dict:
                Parameters to recreate self._model in its current fit status.
        """
        return self._params.copy()

    def _set_params(self, params):
        """Set the parameters of this univariate.

        Args:
            params (dict):
                Parameters to recreate this instance.
        """
        self._params = params.copy()
        if self._is_constant():
            constant = self._extract_constant()
            self._set_constant_value(constant)


class GaussianUnivariate(ScipyModel):
    """Gaussian univariate model."""

    PARAMETRIC = ParametricType.PARAMETRIC
    BOUNDED = BoundedType.UNBOUNDED

    MODEL_CLASS = stats.norm

    def _fit_constant(self, X):
        self._params = {'loc': np.unique(X)[0], 'scale': 0}

    def _fit(self, X):
        self._params = {'loc': np.mean(X), 'scale': np.std(X)}

    def _is_constant(self):
        return self._params['scale'] == 0

    def _extract_constant(self):
        return self._params['loc']
    


DEFAULT_DISTRIBUTION = Univariate
LOGGER = logging.getLogger(__name__)
class GaussianMultivariate(Multivariate):
    """Class for a multivariate distribution that uses the Gaussian copula.

    Args:
        distribution (str or dict):
            Fully qualified name of the class to be used for modeling the marginal
            distributions or a dictionary mapping column names to the fully qualified
            distribution names.
    """

    correlation = None
    columns = None
    univariates = None

    @store_args
    def __init__(self, distribution=DEFAULT_DISTRIBUTION, random_state=None):
        self.random_state = validate_random_state(random_state)
        self.distribution = distribution

    def __repr__(self):
        """Produce printable representation of the object."""
        if self.distribution == DEFAULT_DISTRIBUTION:
            distribution = ''
        elif isinstance(self.distribution, type):
            distribution = f'distribution="{self.distribution.__name__}"'
        else:
            distribution = f'distribution="{self.distribution}"'

        return f'GaussianMultivariate({distribution})'

    def _transform_to_normal(self, X):
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        elif not isinstance(X, pd.DataFrame):
            if len(X.shape) == 1:
                X = [X]

            X = pd.DataFrame(X, columns=self.columns)

        U = []
        for column_name, univariate in zip(self.columns, self.univariates):
            if column_name in X:
                column = X[column_name]
                EPSILON = np.finfo(np.float32).eps
                U.append(univariate.cdf(column.to_numpy()).clip(EPSILON, 1 - EPSILON))

        return stats.norm.ppf(np.column_stack(U))

    @check_valid_values
    def fit(self, X):
        """Compute the distribution for each variable and then its correlation matrix.

        Arguments:
            X (pandas.DataFrame):
                Values of the random variables.
        """
        LOGGER.info('Fitting %s', self)

        # Validate the input data
        X = self._validate_input(X)
        columns, univariates = self._fit_columns(X)

        self.columns = columns
        self.univariates = univariates

        LOGGER.debug('Computing correlation.')
        self.correlation = self._get_correlation(X)
        self.fitted = True
        LOGGER.debug('GaussianMultivariate fitted successfully')

    def _validate_input(self, X):
        """Validate the input data."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        return X

    def _fit_columns(self, X):
        """Fit each column to its distribution."""
        columns = []
        univariates = []
        for column_name, column in X.items():
            distribution = self._get_distribution_for_column(column_name)
            LOGGER.debug('Fitting column %s to %s', column_name, distribution)

            univariate = self._fit_column(column, distribution, column_name)
            columns.append(column_name)
            univariates.append(univariate)

        return columns, univariates

    def _get_distribution_for_column(self, column_name):
        """Retrieve the distribution for a given column name."""
        if isinstance(self.distribution, dict):
            return self.distribution.get(column_name, DEFAULT_DISTRIBUTION)

        return self.distribution

    def _fit_column(self, column, distribution, column_name):
        """Fit a single column to its distribution with exception handling."""
        univariate = get_instance(distribution)
        try:
            univariate.fit(column)
        except Exception as error:
            univariate = self._fit_with_fallback_distribution(
                column, distribution, column_name, error
            )

        return univariate

    def _fit_with_fallback_distribution(self, column, distribution, column_name, error):
        """Fall back to fitting a Gaussian distribution and log the error."""
        log_message = (
            f'Unable to fit to a {distribution} distribution for column {column_name}. '
            'Using a Gaussian distribution instead.'
        )
        LOGGER.info(log_message)
        univariate = GaussianUnivariate()
        univariate.fit(column)
        return univariate

    def _get_correlation(self, X):
        """Compute correlation matrix with transformed data.

        Args:
            X (numpy.ndarray):
                Data for which the correlation needs to be computed.

        Returns:
            numpy.ndarray:
                computed correlation matrix.
        """
        result = self._transform_to_normal(X)
        correlation = pd.DataFrame(data=result).corr().to_numpy()
        correlation = np.nan_to_num(correlation, nan=0.0)
        # If singular, add some noise to the diagonal
        if np.linalg.cond(correlation) > 1.0 / sys.float_info.epsilon:
            EPSILON = np.finfo(np.float32).eps
            correlation = correlation + np.identity(correlation.shape[0]) * EPSILON

        return pd.DataFrame(correlation, index=self.columns, columns=self.columns)

    def probability_density(self, X):
        """Compute the probability density for each point in X.

        Arguments:
            X (pandas.DataFrame):
                Values for which the probability density will be computed.

        Returns:
            numpy.ndarray:
                Probability density values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        transformed = self._transform_to_normal(X)

        return stats.multivariate_normal.pdf(transformed, cov=self.correlation, allow_singular=True)

    def cumulative_distribution(self, X):
        """Compute the cumulative distribution value for each point in X.

        Arguments:
            X (pandas.DataFrame):
                Values for which the cumulative distribution will be computed.

        Returns:
            numpy.ndarray:
                Cumulative distribution values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        transformed = self._transform_to_normal(X)
        return stats.multivariate_normal.cdf(transformed, cov=self.correlation)

    def _get_conditional_distribution(self, conditions):
        """Compute the parameters of a conditional multivariate normal distribution.

        The parameters of the conditioned distribution are computed as specified here:
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions

        Args:
            conditions (pandas.Series):
                Mapping of the column names and column values to condition on.
                The input values have already been transformed to their normal distribution.

        Returns:
            tuple:
                * means (numpy.array):
                    mean values to use for the conditioned multivariate normal.
                * covariance (numpy.array):
                    covariance matrix to use for the conditioned
                  multivariate normal.
                * columns (list):
                    names of the columns that will be sampled conditionally.
        """
        columns2 = conditions.index
        columns1 = self.correlation.columns.difference(columns2)

        sigma11 = self.correlation.loc[columns1, columns1].to_numpy()
        sigma12 = self.correlation.loc[columns1, columns2].to_numpy()
        sigma21 = self.correlation.loc[columns2, columns1].to_numpy()
        sigma22 = self.correlation.loc[columns2, columns2].to_numpy()

        mu1 = np.zeros(len(columns1))
        mu2 = np.zeros(len(columns2))

        sigma12sigma22inv = sigma12 @ np.linalg.inv(sigma22)

        mu_bar = mu1 + sigma12sigma22inv @ (conditions - mu2)
        sigma_bar = sigma11 - sigma12sigma22inv @ sigma21

        return mu_bar, sigma_bar, columns1

    def _get_normal_samples(self, num_rows, conditions):
        """Get random rows in the standard normal space.

        If no conditions are given, the values are sampled from a standard normal
        multivariate.

        If conditions are given, they are transformed to their equivalent standard
        normal values using their marginals and then the values are sampled from
        a standard normal multivariate conditioned on the given condition values.
        """
        if conditions is None:
            covariance = self.correlation
            columns = self.columns
            means = np.zeros(len(columns))
        else:
            conditions = pd.Series(conditions)
            normal_conditions = self._transform_to_normal(conditions)[0]
            normal_conditions = pd.Series(normal_conditions, index=conditions.index)
            means, covariance, columns = self._get_conditional_distribution(normal_conditions)

        samples = np.random.multivariate_normal(means, covariance, size=num_rows)
        return pd.DataFrame(samples, columns=columns)

    @random_state
    def sample(self, num_rows=1, conditions=None):
        """Sample values from this model.

        Argument:
            num_rows (int):
                Number of rows to sample.
            conditions (dict or pd.Series):
                Mapping of the column names and column values to condition on.

        Returns:
            numpy.ndarray:
                Array of shape (n_samples, *) with values randomly
                sampled from this model distribution. If conditions have been
                given, the output array also contains the corresponding columns
                populated with the given values.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()

        samples = self._get_normal_samples(num_rows, conditions)

        output = {}
        for column_name, univariate in zip(self.columns, self.univariates):
            if conditions and column_name in conditions:
                # Use the values that were given as conditions in the original space.
                output[column_name] = np.full(num_rows, conditions[column_name])
            else:
                cdf = stats.norm.cdf(samples[column_name])
                output[column_name] = univariate.percent_point(cdf)

        return pd.DataFrame(data=output)

    def to_dict(self):
        """Return a `dict` with the parameters to replicate this object.

        Returns:
            dict:
                Parameters of this distribution.
        """
        self.check_fit()
        univariates = [univariate.to_dict() for univariate in self.univariates]

        return {
            'correlation': self.correlation.to_numpy().tolist(),
            'univariates': univariates,
            'columns': self.columns,
            'type': get_qualified_name(self),
        }

    @classmethod
    def from_dict(cls, copula_dict):
        """Create a new instance from a parameters dictionary.

        Args:
            params (dict):
                Parameters of the distribution, in the same format as the one
                returned by the ``to_dict`` method.

        Returns:
            Multivariate:
                Instance of the distribution defined on the parameters.
        """
        instance = cls()
        instance.univariates = []
        columns = copula_dict['columns']
        instance.columns = columns

        for parameters in copula_dict['univariates']:
            instance.univariates.append(Univariate.from_dict(parameters))

        correlation = copula_dict['correlation']
        instance.correlation = pd.DataFrame(correlation, index=columns, columns=columns)
        instance.fitted = True

        return instance