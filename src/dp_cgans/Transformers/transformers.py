"""Transformers for categorical data."""

import sys
import logging
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from importlib import import_module

from pandas.api.types import is_datetime64_dtype, is_numeric_dtype
from pandas.core.tools.datetimes import _guess_datetime_format_for_array

from dp_cgans.Transformers.base import BaseTransformer
from dp_cgans.Transformers.base import NullTransformer



LOGGER = logging.getLogger(__name__)


def fill_nan_with_none(data):
    return data.infer_objects().fillna(np.nan).replace([np.nan], [None])

def check_nan_in_transform(data, dtype):
    if pd.isna(data).any().any():
        message = (
            'There are null values in the transformed data. The reversed '
            'transformed data will contain null values'
        )
        is_integer = pd.api.types.is_integer_dtype(dtype)
        if is_integer:
            message += " of type 'float'."
        else:
            message += '.'

        warnings.warn(message)

def try_convert_to_dtype(data, dtype):
    try:
        data = data.astype(dtype)
    except ValueError as error:
        is_integer = pd.api.types.is_integer_dtype(dtype)
        if is_integer:
            data = data.astype(float)
        else:
            raise error

    return data


class OneHotEncoder(BaseTransformer):

    INPUT_SDTYPE = 'categorical'
    SUPPORTED_SDTYPES = ['categorical', 'boolean']
    dummies = None
    _dummy_na = None
    _num_dummies = None
    _dummy_encoded = False
    _indexer = None
    _uniques = None
    dtype = None

    @staticmethod
    def _prepare_data(data):
        """Transform data to appropriate format.

        If data is a valid list or a list of lists, transforms it into an np.array,
        otherwise returns it.

        Args:
            data (pandas.Series or pandas.DataFrame):
                Data to prepare.

        Returns:
            pandas.Series or numpy.ndarray
        """
        if isinstance(data, list):
            data = np.array(data)

        if len(data.shape) > 2:
            raise ValueError('Unexpected format.')
        if len(data.shape) == 2:
            if data.shape[1] != 1:
                raise ValueError('Unexpected format.')

            data = data[:, 0]

        return data

    def _fit(self, data):
        """Fit the transformer to the data.

        Get the pandas `dummies` which will be used later on for OneHotEncoding.

        Args:
            data (pandas.Series or pandas.DataFrame):
                Data to fit the transformer to.
        """
        self.dtype = data.dtype
        data = self._prepare_data(data)

        null = pd.isna(data).to_numpy()
        self._uniques = list(pd.unique(data[~null]))
        self._dummy_na = null.any()
        self._num_dummies = len(self._uniques)
        self._indexer = list(range(self._num_dummies))
        self.dummies = self._uniques.copy()

        if not np.issubdtype(data.dtype.type, np.number):
            self._dummy_encoded = True

        if self._dummy_na:
            self.dummies.append(np.nan)

        self.output_properties = {
            f'value{i}': {'sdtype': 'float', 'next_transformer': None}
            for i in range(len(self.dummies))
        }

    def _transform_helper(self, data):
        if self._dummy_encoded:
            coder = self._indexer
            codes = pd.Categorical(data, categories=self._uniques).codes
        else:
            coder = self._uniques
            codes = data

        rows = len(data)
        dummies = np.broadcast_to(coder, (rows, self._num_dummies))
        coded = np.broadcast_to(codes, (self._num_dummies, rows)).T
        array = (coded == dummies).astype(int)

        if self._dummy_na:
            null = np.zeros((rows, 1), dtype=int)
            null[pd.isna(data)] = 1
            array = np.append(array, null, axis=1)

        return array

    def _transform(self, data):
        """Replace each category with the OneHot vectors.

        Args:
            data (pandas.Series, list or list of lists):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        data = self._prepare_data(data)
        unique_data = {np.nan if pd.isna(x) else x for x in pd.unique(data)}
        unseen_categories = unique_data - {np.nan if pd.isna(x) else x for x in self.dummies}
        if unseen_categories:
            # Select only the first 5 unseen categories to avoid flooding the console.
            examples_unseen_categories = set(list(unseen_categories)[:5])
            warnings.warn(
                f'The data contains {len(unseen_categories)} new categories that were not '
                f'seen in the original data (examples: {examples_unseen_categories}). Creating '
                'a vector of all 0s. If you want to model new categories, '
                'please fit the transformer again with the new data.'
            )

        return self._transform_helper(data)
    


    def _reverse_transform(self, data):
        """Convert float values back to the original categorical values.

        Args:
            data (pd.Series or numpy.ndarray):
                Data to revert.

        Returns:
            pandas.Series
        """
        check_nan_in_transform(data, self.dtype)
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        indices = np.argmax(data, axis=1)
        result = pd.Series(indices).map(self.dummies.__getitem__)
        result = try_convert_to_dtype(result, self.dtype)

        return result
    


########## FloatFormatter 
EPSILON = np.finfo(np.float32).eps
MAX_DECIMALS = sys.float_info.dig
INTEGER_BOUNDS = {
    'Int8': (-(2**7), 2**7 - 1),
    'Int16': (-(2**15), 2**15 - 1),
    'Int32': (-(2**31), 2**31 - 1),
    'Int64': (-(2**63), 2**63 - 1),
    'UInt8': (0, 2**8 - 1),
    'UInt16': (0, 2**16 - 1),
    'UInt32': (0, 2**32 - 1),
    'UInt64': (0, 2**64 - 1),
}


def learn_rounding_digits(data):
    """Learn the number of digits to round data to.

    Args:
        data (pd.Series):
            Data to learn the number of digits to round to.

    Returns:
        int or None:
            Number of digits to round to.
    """
    # check if data has any decimals
    name = data.name
    if str(data.dtype).endswith('[pyarrow]'):
        data = data.to_numpy()
    roundable_data = data[~(np.isinf(data.astype(float)) | pd.isna(data))]

    # Empty dataset
    if len(roundable_data) == 0:
        return None

    if roundable_data.dtype == 'object':
        roundable_data = roundable_data.astype(float)

    # Try to round to fewer digits
    highest_int = int(np.max(np.abs(roundable_data)))
    most_digits = len(str(highest_int)) if highest_int != 0 else 0
    max_decimals = max(0, MAX_DECIMALS - most_digits)
    if (roundable_data == roundable_data.round(max_decimals)).all():
        for decimal in range(max_decimals + 1):
            if (roundable_data == roundable_data.round(decimal)).all():
                return decimal

    # Can't round, not equal after MAX_DECIMALS digits of precision
    LOGGER.info(
        "No rounding scheme detected for column '%s'. Data will not be rounded.",
        name,
    )
    return None



class FloatFormatter(BaseTransformer):
    """Transformer for numerical data.

    This transformer replaces integer values with their float equivalent.
    Non null float values are not modified.

    Null values are replaced using a ``NullTransformer``.

    Args:
        missing_value_replacement (object):
            Indicate what to replace the null values with. If an integer or float is given,
            replace them with the given value. If the strings ``'mean'`` or ``'mode'``
            are given, replace them with the corresponding aggregation and if ``'random'``
            replace each null value with a random value in the data range. Defaults to ``mean``.
         model_missing_values (bool):
            **DEPRECATED** Whether to create a new column to indicate which values were null or
            not. The column will be created only if there are null values. If ``True``, create
            the new column if there are null values. If ``False``, do not create the new column
            even if there are null values. Defaults to ``False``.
        learn_rounding_scheme (bool):
            Whether or not to learn what place to round to based on the data seen during ``fit``.
            If ``True``, the data returned by ``reverse_transform`` will be rounded to that place.
            Defaults to ``False``.
        enforce_min_max_values (bool):
            Whether or not to clip the data returned by ``reverse_transform`` to the min and
            max values seen during ``fit``. Defaults to ``False``.
        computer_representation (dtype):
            Accepts ``'Int8'``, ``'Int16'``, ``'Int32'``, ``'Int64'``, ``'UInt8'``, ``'UInt16'``,
            ``'UInt32'``, ``'UInt64'``, ``'Float'``.
            Defaults to ``'Float'``.
        missing_value_generation (str or None):
            The way missing values are being handled. There are three strategies:

                * ``random``: Randomly generates missing values based on the percentage of
                  missing values.
                * ``from_column``: Creates a binary column that describes whether the original
                  value was missing. Then use it to recreate missing values.
                * ``None``: Do nothing with the missing values on the reverse transform. Simply
                  pass whatever data we get through.
    """

    INPUT_SDTYPE = 'numerical'
    null_transformer = None
    missing_value_replacement = None
    _dtype = None
    _rounding_digits = None
    _min_value = None
    _max_value = None

    def __init__(
        self,
        missing_value_replacement='mean',
        model_missing_values=None,
        learn_rounding_scheme=False,
        enforce_min_max_values=False,
        computer_representation='Float',
        missing_value_generation='random',
    ):
        super().__init__()
        self.missing_value_replacement = missing_value_replacement
        self._set_missing_value_generation(missing_value_generation)
        if model_missing_values is not None:
            self._set_model_missing_values(model_missing_values)

        self.learn_rounding_scheme = learn_rounding_scheme
        self.enforce_min_max_values = enforce_min_max_values
        self.computer_representation = computer_representation

    def _raise_out_of_bounds_error(self, value, name, bound_type, min_bound, max_bound):
        raise ValueError(
            f"The {bound_type} value in column '{name}' is {value}."
            f" All values represented by '{self.computer_representation}'"
            f' must be in the range [{min_bound}, {max_bound}].'
        )

    def _validate_values_within_bounds(self, data):
        if not self.computer_representation.startswith('Float'):
            fractions = data[~data.isna() & (data != (data // 1))]
            if not fractions.empty:
                raise ValueError(
                    f"The column '{data.name}' contains float values {fractions.tolist()}. "
                    f"All values represented by '{self.computer_representation}' must be integers."
                )

            min_value = data.min()
            max_value = data.max()
            min_bound, max_bound = INTEGER_BOUNDS[self.computer_representation]
            if min_value < min_bound:
                self._raise_out_of_bounds_error(
                    min_value, data.name, 'minimum', min_bound, max_bound
                )

            if max_value > max_bound:
                self._raise_out_of_bounds_error(
                    max_value, data.name, 'maximum', min_bound, max_bound
                )



    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit.
        """
        self._validate_values_within_bounds(data)
        self._dtype = data.dtype

        if self.enforce_min_max_values:
            self._min_value = data.min()
            self._max_value = data.max()

        if self.learn_rounding_scheme:
            self._rounding_digits = learn_rounding_digits(data)

        self.null_transformer = NullTransformer(
            self.missing_value_replacement, self.missing_value_generation
        )
        self.null_transformer.fit(data)
        if self.null_transformer.models_missing_values():
            self.output_properties['is_null'] = {
                'sdtype': 'float',
                'next_transformer': None,
            }

    def _transform(self, data):
        """Transform numerical data.

        Integer values are replaced by their float equivalent. Non null float values
        are left unmodified.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        self._validate_values_within_bounds(data)
        data = data.astype(np.float64)
        return self.null_transformer.transform(data)

    def _reverse_transform(self, data):
        """Convert data back into the original format.

        Args:
            data (pd.Series or numpy.ndarray):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()

        data = self.null_transformer.reverse_transform(data)
        if self.enforce_min_max_values:
            data = data.clip(self._min_value, self._max_value)
        elif not self.computer_representation.startswith('Float'):
            min_bound, max_bound = INTEGER_BOUNDS[self.computer_representation]
            data = data.clip(min_bound, max_bound)

        is_integer = pd.api.types.is_integer_dtype(self._dtype)
        np_integer_with_nans = (
            not pd.api.types.is_extension_array_dtype(self._dtype)
            and is_integer
            and pd.isna(data).any()
        )
        if self.learn_rounding_scheme and self._rounding_digits is not None:
            data = data.round(self._rounding_digits)
        elif is_integer:
            data = data.round(0)

        return data.astype(self._dtype if not np_integer_with_nans else 'float64')

    def _set_fitted_parameters(
        self,
        column_name,
        null_transformer,
        rounding_digits=None,
        min_max_values=None,
        dtype='object',
    ):
        """Manually set the parameters on the transformer to get it into a fitted state.

        Args:
            column_name (str):
                The name of the column to use for the transformer.
            null_transformer (NullTransformer):
                A fitted null transformer instance that can be used to generate
                null values for the column.
            min_max_values (Tuple(float) or None):
                None or a tuple containing the (min, max) values for the transformer.
            rounding_digits (int or None):
                The number of digits to round to.
            dtype (str):
                The pandas dtype the reversed data will be converted into.
        """
        self.reset_randomization()
        self.null_transformer = null_transformer
        self.columns = [column_name]
        self.output_columns = [column_name]
        if self.enforce_min_max_values:
            if not min_max_values:
                raise ValueError('Must provide min and max values for this transformer.')

        if min_max_values:
            self._min_value = min(min_max_values)
            self._max_value = max(min_max_values)

        if rounding_digits is not None:
            self._rounding_digits = rounding_digits
            self.learn_rounding_scheme = True

        if self.null_transformer.models_missing_values():
            self.output_columns.append(column_name + '.is_null')

        self._dtype = dtype


##### UniformEnconder


class UniformEncoder(BaseTransformer):
    """Transformer for categorical data.

    This transformer computes a float representative for each one of the categories
    found in the fit data, and then replaces the instances of these categories with
    the corresponding representative.

    The representatives are decided by computing the frequencies of each labels and
    then dividing the ``[0, 1]`` interval according to these frequencies.

    When the transformation is reverted, each value is assigned the category that
    corresponds to the interval it falls in.

    Null values are considered just another category.

    Args:
        order_by (str or None):
            String defining how to order the data before applying the labels. Options are
            'alphabetical', 'numerical' and ``None``. Defaults to ``None``.
    """

    INPUT_SDTYPE = 'categorical'
    SUPPORTED_SDTYPES = ['categorical', 'boolean', 'id', 'text']
    frequencies = None
    intervals = None
    dtype = None

    def __init__(self, order_by=None):
        super().__init__()
        if order_by not in [None, 'alphabetical', 'numerical_value']:
            raise ValueError(
                "order_by must be one of the following values: None, 'numerical_value' or "
                "'alphabetical'"
            )

        self.order_by = order_by

    def _order_categories(self, unique_data):
        nans = pd.isna(unique_data)
        if self.order_by == 'alphabetical':
            # pylint: disable=invalid-unary-operand-type
            if any(map(lambda item: not isinstance(item, str), unique_data[~nans])):  # noqa: C417
                raise ValueError(
                    "The data must be of type string if order_by is 'alphabetical'."
                )
        elif self.order_by == 'numerical_value':
            if not np.issubdtype(unique_data.dtype.type, np.number):
                raise ValueError(
                    "The data must be numerical if order_by is 'numerical_value'."
                )

        if self.order_by is not None:
            unique_data = np.sort(unique_data[~nans])  # pylint: disable=invalid-unary-operand-type
            if nans.any():
                unique_data = np.append(unique_data, [None])

        return unique_data

    @classmethod
    def _get_message_unseen_categories(cls, unseen_categories):
        """Message to raise when there is unseen categories.

        Args:
            unseen_categories (list): list of unseen categories

        Returns:
            message to print
        """
        categories_to_print = ', '.join(str(x) for x in unseen_categories[:3])
        if len(unseen_categories) > 3:
            categories_to_print = f'{categories_to_print}, +{len(unseen_categories) - 3} more'

        return categories_to_print

    @staticmethod
    def _compute_frequencies_intervals(categories, freq):
        """Compute the frequencies and intervals of the categories.

        Args:
            categories (list):
                List of categories.
            freq (list):
                List of frequencies.

        Returns:
            tuple[dict, dict]:
                First dict maps categories to their frequency and the
                second dict maps the categories to their intervals.
        """
        frequencies = dict(zip(categories, freq))
        shift = np.cumsum(np.hstack([0, freq]))
        shift[-1] = 1
        list_int = [[shift[i], shift[i + 1]] for i in range(len(shift) - 1)]
        intervals = dict(zip(categories, list_int))

        return frequencies, intervals

    def _fit(self, data):
        """Fit the transformer to the data.

        Compute the frequencies of each category and use them
        to map the column to a numerical one.

        Args:
            data (pandas.Series):
                Data to fit the transformer to.
        """
        self.dtype = data.dtypes
        data = fill_nan_with_none(data)
        labels = pd.unique(data)
        labels = self._order_categories(labels)
        freq = data.value_counts(normalize=True, dropna=False)
        nan_value = freq[np.nan] if np.nan in freq.index else None
        freq = freq.reindex(labels, fill_value=nan_value).array

        self.frequencies, self.intervals = self._compute_frequencies_intervals(labels, freq)

    def _set_fitted_parameters(self, column_name, intervals, dtype='object'):
        """Manually set the parameters on the transformer to get it into a fitted state.

        Args:
            column_name (str):
                The name of the column for this transformer.
            intervals (dict[str, tuple]):
                A dictionary mapping categories to the interval in the range [0, 1]
                it should map to.
            dtype (str, optional):
                The dtype to convert the reverse transformed data back to. Defaults to 'object'.
        """
        self.reset_randomization()
        self.columns = [column_name]
        self.output_columns = [column_name]
        self.intervals = intervals
        self.dtype = dtype

    def _transform(self, data):
        """Map the category to a continuous value.

        This value is sampled from a uniform distribution
        with boudaries defined by the frequencies.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns:
            pandas.Series
        """
        data_with_none = fill_nan_with_none(data)
        unseen_indexes = ~(data_with_none.isin(self.frequencies))
        if unseen_indexes.any():
            # Keep the 3 first unseen categories
            unseen_categories = list(data.loc[unseen_indexes].unique())
            categories_to_print = self._get_message_unseen_categories(unseen_categories)
            warnings.warn(
                f"The data in column '{self.get_input_column()}' contains new categories "
                f"that did not appear during 'fit' ({categories_to_print}). Assigning "
                'them random values. If you want to model new categories, '
                "please fit the data again using 'fit'.",
                category=UserWarning,
            )

            choices = list(self.frequencies.keys())
            size = unseen_indexes.size
            data_with_none[unseen_indexes] = np.random.choice(choices, size=size)

        def map_labels(label):
            return np.random.uniform(self.intervals[label][0], self.intervals[label][1])

        return data_with_none.map(map_labels).astype(float)

    def _reverse_transform(self, data):
        """Convert float values back to the original categorical values.

        Args:
            data (pandas.Series):
                Data to revert.

        Returns:
            pandas.Series
        """
        check_nan_in_transform(data, self.dtype)
        data = data.clip(0, 1)
        bins = [0]
        labels = []
        nan_name = 'NaN'
        while nan_name in self.intervals.keys():
            nan_name += '_'

        for key, interval in self.intervals.items():
            bins.append(interval[1])
            if pd.isna(key):
                labels.append(nan_name)
            else:
                labels.append(key)

        result = pd.cut(data, bins=bins, labels=labels, include_lowest=True)
        if nan_name in result.cat.categories:
            result = result.cat.remove_categories(nan_name)

        result = try_convert_to_dtype(result, self.dtype)

        return result


########### BinaryEncoder


class BinaryEncoder(BaseTransformer):
    """Transformer for boolean data.

    This transformer replaces boolean values with their integer representation
    transformed to float.

    Null values are replaced using a ``NullTransformer``.

    Args:
        missing_value_replacement (object):
            Indicate what to replace the null values with. If the string ``'mode'`` is given,
            replace them with the most common value.
            Defaults to ``mode``.
        model_missing_values (bool):
            **DEPRECATED** Whether to create a new column to indicate which values were null or
            not. The column will be created only if there are null values. If ``True``, create
            the new column if there are null values. If ``False``, do not create the new column
            even if there are null values. Defaults to ``False``.
        missing_value_generation (str or None):
            The way missing values are being handled. There are three strategies:

                * ``random``: Randomly generates missing values based on the percentage of
                  missing values.
                * ``from_column``: Creates a binary column that describes whether the original
                  value was missing. Then use it to recreate missing values.
                * ``None``: Do nothing with the missing values on the reverse transform. Simply
                  pass whatever data we get through.
    """

    INPUT_SDTYPE = 'boolean'
    null_transformer = None

    def __init__(
        self,
        missing_value_replacement='mode',
        model_missing_values=None,
        missing_value_generation='random',
    ):
        super().__init__()
        self.missing_value_replacement = missing_value_replacement
        self._set_missing_value_generation(missing_value_generation)
        if model_missing_values is not None:
            self._set_model_missing_values(model_missing_values)

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit to.
        """
        self.null_transformer = NullTransformer(
            self.missing_value_replacement, self.missing_value_generation
        )
        self.null_transformer.fit(data)
        if self.null_transformer.models_missing_values():
            self.output_properties['is_null'] = {
                'sdtype': 'float',
                'next_transformer': None,
            }

    def _transform(self, data):
        """Transform boolean to float.

        The boolean values will be replaced by the corresponding integer
        representations as float values.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns:
            np.ndarray
        """
        data = pd.to_numeric(data, errors='coerce')
        return self.null_transformer.transform(data).astype(float)

    def _reverse_transform(self, data):
        """Transform float values back to the original boolean values.

        Args:
            data (pandas.DataFrame or pandas.Series):
                Data to revert.

        Returns:
            pandas.Series:
                Reverted data.
        """
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()

        data = self.null_transformer.reverse_transform(data)
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                data = data[:, 0]

            data = pd.Series(data)

        isna = data.isna()
        data = np.round(data).clip(0, 1).astype('boolean').astype('object')
        data[isna] = np.nan

        return data

    def _set_fitted_parameters(self, column_name, null_transformer):
        """Manually set the parameters on the transformer to get it into a fitted state.

        Args:
            column_name (str):
                The name of the column to use for the transformer.
            null_transformer (NullTransformer):
                A fitted null transformer instance that can be used to generate
                null values for the column.
        """
        self.reset_randomization()
        self.columns = [column_name]
        self.output_columns = [column_name]
        self.null_transformer = null_transformer
        if self.null_transformer.models_missing_values():
            self.output_columns.append(column_name + '.is_null')


########## UnixTimestampEncoder


class UnixTimestampEncoder(BaseTransformer):
    """Transformer for datetime data.

    This transformer replaces datetime values with an integer timestamp
    transformed to float.

    Null values are replaced using a ``NullTransformer``.

    Args:
        missing_value_replacement (object):
            Indicate what to replace the null values with. If the strings ``'mean'`` or ``'mode'``
            are given, replace them with the corresponding aggregation, if ``'random'``, use
            random values from the dataset to fill the nan values.
            Defaults to ``mean``.
        model_missing_values (bool):
            **DEPRECATED** Whether to create a new column to indicate which values were null or
            not. The column will be created only if there are null values. If ``True``, create
            the new column if there are null values. If ``False``, do not create the new column
            even if there are null values. Defaults to ``False``.
        datetime_format (str):
            The strftime to use for parsing time. For more information, see
            https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior.
        missing_value_generation (str or None):
            The way missing values are being handled. There are three strategies:

                * ``random``: Randomly generates missing values based on the percentage of
                  missing values.
                * ``from_column``: Creates a binary column that describes whether the original
                  value was missing. Then use it to recreate missing values.
                * ``None``: Do nothing with the missing values on the reverse transform. Simply
                  pass whatever data we get through.
        enforce_min_max_values (bool):
            Whether or not to clip the data returned by ``reverse_transform`` to the min and
            max values seen during ``fit``. Defaults to ``False``.
    """

    INPUT_SDTYPE = 'datetime'
    null_transformer = None
    _min_value = None
    _max_value = None

    def __init__(
        self,
        missing_value_replacement='mean',
        model_missing_values=None,
        datetime_format=None,
        missing_value_generation='random',
        enforce_min_max_values=False,
    ):
        super().__init__()
        self.missing_value_replacement = missing_value_replacement
        self._set_missing_value_generation(missing_value_generation)
        self.enforce_min_max_values = enforce_min_max_values
        if model_missing_values is not None:
            self._set_model_missing_values(model_missing_values)

        self.datetime_format = datetime_format
        self._dtype = None

    def _convert_to_datetime(self, data):
        """Convert datetime column into datetime dtype.

        Convert the datetime column to datetime dtype using the ``datetime_format``.
        All non-numeric columns will automatically be cast to datetimes. Numeric columns
        with a ``datetime_format`` will be treated as strings and cast to datetime. Numeric
        columns without a ``datetime_format`` will be treated as already converted datetimes.

        Args:
            data (pandas.Series):
                The datetime column.

        Raises:
            - ``TypeError`` if data cannot be converted to datetime.
            - ``ValueError`` if data does not match the specified datetime format

        Returns:
            pandas.Series:
                The datetime column converted to the datetime dtype.
        """
        if self.datetime_format or not is_numeric_dtype(data):
            try:
                pandas_datetime_format = None
                if self.datetime_format:
                    pandas_datetime_format = self.datetime_format.replace('%-', '%')

                data = pd.to_datetime(data, format=pandas_datetime_format)

            except ValueError as error:
                if 'Unknown string' in str(error) or 'Unknown datetime string' in str(error):
                    message = 'Data must be of dtype datetime, or castable to datetime.'
                    raise TypeError(message) from None

                raise ValueError('Data does not match specified datetime format.') from None

        return data

    def _transform_helper(self, datetimes):
        """Transform datetime values to integer."""
        datetimes = self._convert_to_datetime(datetimes)
        nulls = datetimes.isna()
        integers = pd.to_numeric(datetimes, errors='coerce').to_numpy().astype(np.float64)
        integers[nulls] = np.nan
        transformed = pd.Series(integers)

        return transformed

    def _reverse_transform_helper(self, data):
        """Transform integer values back into datetimes."""
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()

        data = self.null_transformer.reverse_transform(data)
        data = np.round(data.astype(np.float64))
        return data

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit the transformer to.
        """
        self._dtype = data.dtype
        if self.datetime_format is None:
            datetime_array = data[data.notna()].astype(str).to_numpy()
            self.datetime_format = _guess_datetime_format_for_array(datetime_array)

        transformed = self._transform_helper(data)
        if self.enforce_min_max_values:
            self._min_value = transformed.min()
            self._max_value = transformed.max()

        self.null_transformer = NullTransformer(
            self.missing_value_replacement, self.missing_value_generation
        )
        self.null_transformer.fit(transformed)
        if self.null_transformer.models_missing_values():
            self.output_properties['is_null'] = {
                'sdtype': 'float',
                'next_transformer': None,
            }

    def _set_fitted_parameters(
        self, column_name, null_transformer, min_max_values=None, dtype='object'
    ):
        """Manually set the parameters on the transformer to get it into a fitted state.

        Args:
            column_name (str):
                The name of the column for this transformer.
            null_transformer (NullTransformer):
                A fitted null transformer instance that can be used to generate
                null values for the column.
            min_max_values (tuple or None):
                None or a tuple containing the (min, max) values for the transformer.
                Should be used to set self._min_value and self._max_value and must be
                provided if self.enforce_min_max_values is True.
                Defaults to None.
            dtype (str, optional):
                The dtype to convert the reverse transformed data back to. Defaults to 'object'.
        """
        self.reset_randomization()
        self.columns = [column_name]
        self.output_columns = [column_name]
        self._dtype = dtype

        if self.enforce_min_max_values and not min_max_values:
            raise ValueError('Must provide min and max values for this transformer.')

        if min_max_values:
            self._min_value = min_max_values[0]
            self._max_value = min_max_values[1]

        self.null_transformer = null_transformer
        if self.null_transformer.models_missing_values():
            self.output_columns.append(column_name + '.is_null')

    def _transform(self, data):
        """Transform datetime values to float values.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        data = self._transform_helper(data)
        return self.null_transformer.transform(data)

    def _reverse_transform(self, data):
        """Convert float values back to datetimes.

        Args:
            data (pandas.Series or numpy.ndarray):
                Data to transform.

        Returns:
            pandas.Series
        """
        data = self._reverse_transform_helper(data)
        if self.enforce_min_max_values:
            data = data.clip(self._min_value, self._max_value)

        datetime_data = pd.to_datetime(data)
        if self.datetime_format:
            if is_datetime64_dtype(self._dtype) and '.%f' not in self.datetime_format:
                datetime_data = pd.to_datetime(
                    datetime_data.dt.strftime(self.datetime_format),
                    format=self.datetime_format,
                )
            else:
                datetime_data = datetime_data.dt.strftime(self.datetime_format).astype(self._dtype)
        elif is_numeric_dtype(self._dtype):
            datetime_data = pd.to_numeric(datetime_data.astype('object'), errors='coerce')
            datetime_data = datetime_data.astype(self._dtype)

        return datetime_data

###### ClusterBasedNormalizer
class ClusterBasedNormalizer(FloatFormatter):
    """Transformer for numerical data using a Bayesian Gaussian Mixture Model.

    This transformation takes a numerical value and transforms it using a Bayesian GMM
    model. It generates two outputs, a discrete value which indicates the selected
    'component' of the GMM and a continuous value which represents the normalized value
    based on the mean and std of the selected component.

    Args:
        model_missing_values (bool):
            **DEPRECATED** Whether to create a new column to indicate which values were null or
            not. The column will be created only if there are null values. If ``True``, create
            the new column if there are null values. If ``False``, do not create the new column
            even if there are null values. Defaults to ``False``.
        learn_rounding_scheme (bool):
            Whether or not to learn what place to round to based on the data seen during ``fit``.
            If ``True``, the data returned by ``reverse_transform`` will be rounded to that place.
            Defaults to ``False``.
        enforce_min_max_values (bool):
            Whether or not to clip the data returned by ``reverse_transform`` to the min and
            max values seen during ``fit``. Defaults to ``False``.
        max_clusters (int):
            The maximum number of mixture components. Depending on the data, the model may select
            fewer components (based on the ``weight_threshold``).
            Defaults to 10.
        weight_threshold (int, float):
            The minimum value a component weight can take to be considered a valid component.
            ``weights_`` under this value will be ignored.
            Defaults to 0.005.
        missing_value_generation (str or None):
            The way missing values are being handled. There are three strategies:

                * ``random``: Randomly generates missing values based on the percentage of
                  missing values.
                * ``from_column``: Creates a binary column that describes whether the original
                  value was missing. Then use it to recreate missing values.
                * ``None``: Do nothing with the missing values on the reverse transform. Simply
                  pass whatever data we get through.

    Attributes:
        _bgm_transformer:
            An instance of sklearn`s ``BayesianGaussianMixture`` class.
        valid_component_indicator:
            An array indicating the valid components. If the weight of a component is greater
            than the ``weight_threshold``, it's indicated with True, otherwise it's set to False.
    """

    STD_MULTIPLIER = 4
    _bgm_transformer = None
    valid_component_indicator = None

    def __init__(
        self,
        model_missing_values=None,
        learn_rounding_scheme=False,
        enforce_min_max_values=False,
        max_clusters=10,
        weight_threshold=0.005,
        missing_value_generation='random',
    ):
        # Using missing_value_replacement='mean' as the default instead of random
        # as this may lead to different outcomes in certain synthesizers
        # affecting the synthesizers directly and this is out of scope for now.
        super().__init__(
            model_missing_values=model_missing_values,
            missing_value_generation=missing_value_generation,
            missing_value_replacement='mean',
            learn_rounding_scheme=learn_rounding_scheme,
            enforce_min_max_values=enforce_min_max_values,
        )
        self.max_clusters = max_clusters
        self.weight_threshold = weight_threshold
        self.output_properties = {
            'normalized': {'sdtype': 'float', 'next_transformer': None},
            'component': {'sdtype': 'categorical', 'next_transformer': None},
        }

    def _get_current_random_seed(self):
        if self.random_states:
            return self.random_states['fit'].get_state()[1][0]

        return 0

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit to.
        """
        sm = import_module('sklearn.mixture')

        self._bgm_transformer = sm.BayesianGaussianMixture(
            n_components=self.max_clusters,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001,
            random_state=self._get_current_random_seed(),
        )

        super()._fit(data)
        data = super()._transform(data)
        if data.ndim > 1:
            data = data[:, 0]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._bgm_transformer.fit(data.reshape(-1, 1))

        self.valid_component_indicator = self._bgm_transformer.weights_ > self.weight_threshold

