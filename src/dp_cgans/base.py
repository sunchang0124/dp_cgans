"""Base Class for tabular models."""

import logging
import pickle
import uuid
import warnings

import numpy as np
import pandas as pd

from dp_cgans.functions.table import Table

LOGGER = logging.getLogger(__name__)
COND_IDX = str(uuid.uuid4())

############################


"""BaseTransformer module."""

import abc
import contextlib
import hashlib
import inspect
from functools import wraps

############################


class BaseTabularModel:
    """Base class for all the tabular models.

    The ``BaseTabularModel`` class defines the common API that all the
    TabularModels need to implement, as well as common functionality.

    Args:
        field_names (list[str]):
            List of names of the fields that need to be modeled
            and included in the generated output data. Any additional
            fields found in the data will be ignored and will not be
            included in the generated output.
            If ``None``, all the fields found in the data are used.
        field_types (dict[str, dict]):
            Dictinary specifying the data types and subtypes
            of the fields that will be modeled. Field types and subtypes
            combinations must be compatible with the SDV Metadata Schema.
        field_transformers (dict[str, str]):
            Dictinary specifying which transformers to use for each field.
            Available transformers are:

                * ``integer``: Uses a ``NumericalTransformer`` of dtype ``int``.
                * ``float``: Uses a ``NumericalTransformer`` of dtype ``float``.
                * ``categorical``: Uses a ``CategoricalTransformer`` without gaussian noise.
                * ``categorical_fuzzy``: Uses a ``CategoricalTransformer`` adding gaussian noise.
                * ``one_hot_encoding``: Uses a ``OneHotEncodingTransformer``.
                * ``label_encoding``: Uses a ``LabelEncodingTransformer``.
                * ``boolean``: Uses a ``BooleanTransformer``.
                * ``datetime``: Uses a ``DatetimeTransformer``.

        anonymize_fields (dict[str, str]):
            Dict specifying which fields to anonymize and what faker
            category they belong to.
        primary_key (str):
            Name of the field which is the primary key of the table.
        constraints (list[Constraint, dict]):
            List of Constraint objects or dicts.
        table_metadata (dict or metadata.Table):
            Table metadata instance or dict representation.
            If given alongside any other metadata-related arguments, an
            exception will be raised.
            If not given at all, it will be built using the other
            arguments or learned from the data.
        rounding (int, str or None):
            Define rounding scheme for ``NumericalTransformer``. If set to an int, values
            will be rounded to that number of decimal places. If ``None``, values will not
            be rounded. If set to ``'auto'``, the transformer will round to the maximum number
            of decimal places detected in the fitted data. Defaults to ``'auto'``.
        min_value (int, str or None):
            Specify the minimum value the ``NumericalTransformer`` should use. If an integer
            is given, sampled data will be greater than or equal to it. If the string ``'auto'``
            is given, the minimum will be the minimum value seen in the fitted data. If ``None``
            is given, there won't be a minimum. Defaults to ``'auto'``.
        max_value (int, str or None):
            Specify the maximum value the ``NumericalTransformer`` should use. If an integer
            is given, sampled data will be less than or equal to it. If the string ``'auto'``
            is given, the maximum will be the maximum value seen in the fitted data. If ``None``
            is given, there won't be a maximum. Defaults to ``'auto'``.
    """

    _DTYPE_TRANSFORMERS = None

    _metadata = None

    def __init__(self, field_names=None, field_types=None, field_transformers=None,
                 anonymize_fields=None, primary_key=None, constraints=None, table_metadata=None,
                 rounding='auto', min_value='auto', max_value='auto'):
        if table_metadata is None:
            self._metadata = Table(
                field_names=field_names,
                primary_key=primary_key,
                field_types=field_types,
                field_transformers=field_transformers,
                anonymize_fields=anonymize_fields,
                constraints=constraints,
                dtype_transformers=self._DTYPE_TRANSFORMERS,
                rounding=rounding,
                min_value=min_value,
                max_value=max_value
            )
            self._metadata_fitted = False
        else:
            for arg in (field_names, primary_key, field_types, anonymize_fields, constraints):
                if arg:
                    raise ValueError(
                        'If table_metadata is given {} must be None'.format(arg.__name__))

            if isinstance(table_metadata, dict):
                table_metadata = Table.from_dict(table_metadata)

            table_metadata._dtype_transformers.update(self._DTYPE_TRANSFORMERS)

            self._metadata = table_metadata
            self._metadata_fitted = table_metadata.fitted

    def fit(self, data):
        LOGGER.debug('Fitting %s to table %s; shape: %s', self.__class__.__name__,
                     self._metadata.name, data.shape)
        if not self._metadata_fitted:
            self._metadata.fit(data)

        self._num_rows = len(data)

        LOGGER.debug('Transforming table %s; shape: %s', self._metadata.name, data.shape)
        transformed = self._metadata.transform(data)

        if self._metadata.get_dtypes(ids=False):
            LOGGER.debug(
                'Fitting %s model to table %s', self.__class__.__name__, self._metadata.name)
            self._fit(transformed)

    def get_metadata(self):
        return self._metadata

    @staticmethod
    def _filter_conditions(sampled, conditions, float_rtol):
        for column, value in conditions.items():
            column_values = sampled[column]
            if column_values.dtype.kind == 'f':
                distance = value * float_rtol
                sampled = sampled[np.abs(column_values - value) < distance]
                sampled[column] = value
            else:
                sampled = sampled[column_values == value]

        return sampled

    def _sample_rows(self, num_rows, conditions=None, transformed_conditions=None,
                     float_rtol=0.1, previous_rows=None):
        """Sample rows with the given conditions.

        Input conditions is taken both in the raw input format, which will be used
        for filtering during the reject-sampling loop, and already transformed
        to the model format, which will be passed down to the model if it supports
        conditional sampling natively.

        If condition columns are float values, consider a match anything that
        is closer than the given ``float_rtol`` and then make the value exact.

        If the model does not have any data columns, the result of this call
        is a dataframe of the requested length with no columns in it.

        Args:
            num_rows (int):
                Number of rows to sample.
            conditions (dict):
                The dictionary of conditioning values in the original format.
            transformed_conditions (dict):
                The dictionary of conditioning values transformed to the model format.
            float_rtol (float):
                Maximum tolerance when considering a float match.
            previous_rows (pandas.DataFrame):
                Valid rows sampled in the previous iterations.

        Returns:
            tuple:
                * pandas.DataFrame:
                    Rows from the sampled data that match the conditions.
                * int:
                    Number of rows that are considered valid.
        """
        if self._metadata.get_dtypes(ids=False):
            if conditions is None:
                sampled = self._sample(num_rows)
            else:
                try:
                    sampled = self._sample(num_rows, transformed_conditions)
                except NotImplementedError:
                    sampled = self._sample(num_rows)

            sampled = self._metadata.reverse_transform(sampled)

            if previous_rows is not None:
                sampled = previous_rows.append(sampled, ignore_index=True)

            sampled = self._metadata.filter_valid(sampled)

            if conditions is not None:
                sampled = self._filter_conditions(sampled, conditions, float_rtol)

            num_valid = len(sampled)

            return sampled, num_valid

        else:
            sampled = pd.DataFrame(index=range(num_rows))
            sampled = self._metadata.reverse_transform(sampled)
            return sampled, num_rows

    def _sample_batch(self, num_rows=None, max_retries=100, max_rows_multiplier=10,
                      conditions=None, transformed_conditions=None, float_rtol=0.01):
        """Sample a batch of rows with the given conditions.

        This will enter a reject-sampling loop in which rows will be sampled until
        all of them are valid and match the requested conditions. If `max_retries`
        is exceeded, it will return as many rows as it has sampled, which may be less
        than the target number of rows.

        Input conditions is taken both in the raw input format, which will be used
        for filtering during the reject-sampling loop, and already transformed
        to the model format, which will be passed down to the model if it supports
        conditional sampling natively.

        If condition columns are float values, consider a match anything that is
        relatively closer than the given ``float_rtol`` and then make the value exact.

        If the model does not have any data columns, the result of this call
        is a dataframe of the requested length with no columns in it.

        Args:
            num_rows (int):
                Number of rows to sample. If not given the model
                will generate as many rows as there were in the
                data passed to the ``fit`` method.
            max_retries (int):
                Number of times to retry sampling discarded rows.
                Defaults to 100.
            max_rows_multiplier (int):
                Multiplier to use when computing the maximum number of rows
                that can be sampled during the reject-sampling loop.
                The maximum number of rows that are sampled at each iteration
                will be equal to this number multiplied by the requested num_rows.
                Defaults to 10.
            conditions (dict):
                The dictionary of conditioning values in the original input format.
            transformed_conditions (dict):
                The dictionary of conditioning values transformed to the model format.
            float_rtol (float):
                Maximum tolerance when considering a float match.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        sampled, num_valid = self._sample_rows(
            num_rows, conditions, transformed_conditions, float_rtol)

        counter = 0
        total_sampled = num_rows
        while num_valid < num_rows:
            if counter >= max_retries:
                break

            remaining = num_rows - num_valid
            valid_probability = (num_valid + 1) / (total_sampled + 1)
            max_rows = num_rows * max_rows_multiplier
            num_to_sample = min(int(remaining / valid_probability), max_rows)
            total_sampled += num_to_sample

            LOGGER.info('%s valid rows remaining. Resampling %s rows', remaining, num_to_sample)
            sampled, num_valid = self._sample_rows(
                num_to_sample, conditions, transformed_conditions, float_rtol, sampled
            )

            counter += 1

        return sampled.head(min(len(sampled), num_rows))

    def _make_conditions_df(self, conditions, num_rows):
        """Transform `conditions` into a dataframe.

        Args:
            conditions (pd.DataFrame, dict or pd.Series):
                If this is a dictionary/Series which maps column names to the column
                value, then this method generates `num_rows` samples, all of
                which are conditioned on the given variables. If this is a DataFrame,
                then it generates an output DataFrame such that each row in the output
                is sampled conditional on the corresponding row in the input.
            num_rows (int):
                Number of rows to sample. If a conditions dataframe is given, this must
                either be ``None`` or match the length of the ``conditions`` dataframe.

        Returns:
            pandas.DataFrame:
                `conditions` as a dataframe.
        """
        if isinstance(conditions, pd.Series):
            conditions = pd.DataFrame([conditions] * num_rows)

        elif isinstance(conditions, dict):
            try:
                conditions = pd.DataFrame(conditions)
            except ValueError:
                conditions = pd.DataFrame([conditions] * num_rows)

        elif not isinstance(conditions, pd.DataFrame):
            raise TypeError('`conditions` must be a dataframe, a dictionary or a pandas series.')

        elif num_rows is not None and len(conditions) != num_rows:
            raise ValueError(
                'If `conditions` is a `DataFrame`, `num_rows` must be `None` or match its lenght.')

        return conditions.copy()

    def _conditionally_sample_rows(self, dataframe, max_retries, max_rows_multiplier,
                                   condition, transformed_condition, float_rtol,
                                   graceful_reject_sampling):
        num_rows = len(dataframe)
        sampled_rows = self._sample_batch(
            num_rows,
            max_retries,
            max_rows_multiplier,
            condition,
            transformed_condition,
            float_rtol
        )
        num_sampled_rows = len(sampled_rows)

        if num_sampled_rows < num_rows:
            # Didn't get enough rows.
            if len(sampled_rows) == 0:
                error = 'No valid rows could be generated with the given conditions.'
                raise ValueError(error)

            elif not graceful_reject_sampling:
                error = f'Could not get enough valid rows within {max_retries} trials.'
                raise ValueError(error)

            else:
                warnings.warn(f'Only {len(sampled_rows)} rows could '
                     f'be sampled within {max_retries} trials.')

        if len(sampled_rows) > 0:
            sampled_rows[COND_IDX] = dataframe[COND_IDX].values[:len(sampled_rows)]

        return sampled_rows

    def sample(self, num_rows=None, max_retries=100, max_rows_multiplier=10,
               conditions=None, float_rtol=0.01, graceful_reject_sampling=False):
        """Sample rows from this table.

        Args:
            num_rows (int):
                Number of rows to sample. If not given the model
                will generate as many rows as there were in the
                data passed to the ``fit`` method.
            max_retries (int):
                Number of times to retry sampling discarded rows.
                Defaults to 100.
            max_rows_multiplier (int):
                Multiplier to use when computing the maximum number of rows
                that can be sampled during the reject-sampling loop.
                The maximum number of rows that are sampled at each iteration
                will be equal to this number multiplied by the requested num_rows.
                Defaults to 10.
            conditions (pd.DataFrame, dict or pd.Series):
                If this is a dictionary/Series which maps column names to the column
                value, then this method generates `num_rows` samples, all of
                which are conditioned on the given variables. If this is a DataFrame,
                then it generates an output DataFrame such that each row in the output
                is sampled conditional on the corresponding row in the input.
            float_rtol (float):
                Maximum tolerance when considering a float match. This is the maximum
                relative distance at which a float value will be considered a match
                when performing reject-sampling based conditioning. Defaults to 0.01.
            graceful_reject_sampling (bool):
                If `False` raises a `ValueError` if not enough valid rows could be sampled
                within `max_retries` trials. If `True` prints a warning and returns
                as many rows as it was able to sample within `max_retries`.
                Defaults to False.

        Returns:
            pandas.DataFrame:
                Sampled data.

        Raises:
            ConstraintsNotMetError:
                If the conditions are not valid for the given constraints.
            ValueError:
                If any of the following happens:
                    * any of the conditions' columns are not valid.
                    * `graceful_reject_sampling` is `False` and not enough valid rows could be
                      sampled within `max_retries` trials.
                    * no rows could be generated.
        """
        if conditions is None:
            num_rows = num_rows or self._num_rows
            return self._sample_batch(num_rows, max_retries, max_rows_multiplier)

        # convert conditions to dataframe
        conditions = self._make_conditions_df(conditions, num_rows)

        # validate columns
        for column in conditions.columns:
            if column not in self._metadata.get_fields():
                raise ValueError(f'Invalid column name `{column}`')

        try:
            transformed_conditions = self._metadata.transform(conditions, on_missing_column='drop')
        except Exception as cnme:
            cnme.message = 'Passed conditions are not valid for the given constraints'
            raise

        condition_columns = list(conditions.columns)
        transformed_columns = list(transformed_conditions.columns)
        conditions.index.name = COND_IDX
        conditions.reset_index(inplace=True)
        transformed_conditions.index.name = COND_IDX
        transformed_conditions.reset_index(inplace=True)
        grouped_conditions = conditions.groupby(condition_columns)

        # sample
        all_sampled_rows = list()

        for group, dataframe in grouped_conditions:
            if not isinstance(group, tuple):
                group = [group]

            condition_indices = dataframe[COND_IDX]
            condition = dict(zip(condition_columns, group))
            if len(transformed_columns) == 0:
                sampled_rows = self._conditionally_sample_rows(
                    dataframe,
                    max_retries,
                    max_rows_multiplier,
                    condition,
                    None,
                    float_rtol,
                    graceful_reject_sampling
                )
                all_sampled_rows.append(sampled_rows)
            else:
                transformed_conditions_in_group = transformed_conditions.loc[condition_indices]
                transformed_groups = transformed_conditions_in_group.groupby(transformed_columns)
                for transformed_group, transformed_dataframe in transformed_groups:
                    if not isinstance(transformed_group, tuple):
                        transformed_group = [transformed_group]

                    transformed_condition = dict(zip(transformed_columns, transformed_group))
                    sampled_rows = self._conditionally_sample_rows(
                        transformed_dataframe,
                        max_retries,
                        max_rows_multiplier,
                        condition,
                        transformed_condition,
                        float_rtol,
                        graceful_reject_sampling
                    )
                    all_sampled_rows.append(sampled_rows)

        all_sampled_rows = pd.concat(all_sampled_rows)
        all_sampled_rows = all_sampled_rows.set_index(COND_IDX)
        all_sampled_rows.index.name = conditions.index.name
        all_sampled_rows = all_sampled_rows.sort_index()
        all_sampled_rows = self._metadata.make_ids_unique(all_sampled_rows)

        return all_sampled_rows

    def _get_parameters(self):
        raise ValueError()

    def get_parameters(self):
        if self._metadata.get_dtypes(ids=False):
            parameters = self._get_parameters()
        else:
            parameters = {}

        parameters['num_rows'] = self._num_rows
        return parameters

    def _set_parameters(self, parameters):
        raise ValueError()

    def set_parameters(self, parameters):
        num_rows = parameters.pop('num_rows')
        self._num_rows = 0 if pd.isnull(num_rows) else max(0, int(round(num_rows)))

        if self._metadata.get_dtypes(ids=False):
            self._set_parameters(parameters)

    def save(self, path):
        """Save this model instance to the given path using pickle.

        Args:
            path (str):
                Path where the SDV instance will be serialized.
        """
        with open(path, 'wb') as output:
            pickle.dump(self, output)

    def xai_discriminator(self, data_samples):
        discriminator_predict_score = self._xai_discriminator(data_samples)
        return discriminator_predict_score

    @classmethod
    def load(cls, path):
        """Load a TabularModel instance from a given path.

        Args:
            path (str):
                Path from which to load the instance.

        Returns:
            TabularModel:
                The loaded tabular model.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)




############################

@contextlib.contextmanager

def set_random_states(random_states, method_name, set_model_random_state):

    original_np_state = np.random.get_state()
    random_np_state = random_states[method_name]
    np.random.set_state(random_np_state.get_state())

    try:
        yield
    finally:
        current_np_state = np.random.RandomState()
        current_np_state.set_state(np.random.get_state())
        set_model_random_state(current_np_state, method_name)

        np.random.set_state(original_np_state)


def random_state(function):
    """Set the random state before calling the function.

    Args:
        function (Callable):
            The function to wrap around.
    """

    @wraps(function)
    def wrapper(self, *args, **kwargs):
        if self.random_states is None:
            return function(self, *args, **kwargs)

        method_name = function.__name__
        with set_random_states(self.random_states, method_name, self.set_random_state):
            return function(self, *args, **kwargs)

    return wrapper


class BaseTransformer:

    INPUT_SDTYPE = None
    SUPPORTED_SDTYPES = None
    IS_GENERATOR = None
    INITIAL_FIT_STATE = np.random.RandomState(42)

    columns = None
    column_prefix = None
    output_columns = None
    random_seed = 42
    missing_value_replacement = None
    missing_value_generation = None

    def __init__(self):
        self.output_properties = {None: {'sdtype': 'float', 'next_transformer': None}}
        self.random_states = {
            'fit': self.INITIAL_FIT_STATE,
            'transform': None,
            'reverse_transform': None,
        }

    def set_random_state(self, state, method_name):
        """Set the random state for a transformer.

        Args:
            state (numpy.random.RandomState):
                The numpy random state to set.
            method_name (str):
                The method to set it for.
        """
        if method_name not in self.random_states:
            raise ValueError(
                "'method_name' must be one of 'fit', 'transform' or 'reverse_transform'."
            )

        self.random_states[method_name] = state

    def reset_randomization(self):
        """Reset the random state for ``reverse_transform``."""
        self.random_states = {
            'fit': self.INITIAL_FIT_STATE,
            'transform': np.random.RandomState(self.random_seed),
            'reverse_transform': np.random.RandomState(self.random_seed + 1),
        }

    @property
    def model_missing_values(self):
        """Return whether or not a new column is being used to model missing values."""
        warnings.warn(
            "Future versions of RDT will not support the 'model_missing_values' parameter. "
            "Please switch to using the 'missing_value_generation' parameter instead.",
            FutureWarning,
        )
        return self.missing_value_generation == 'from_column'

    def _set_missing_value_generation(self, missing_value_generation):
        if missing_value_generation not in (None, 'from_column', 'random'):
            raise NotImplementedError(
            "'missing_value_generation' must be one of the following values: "
                "None, 'from_column' or 'random'."
        )

        self.missing_value_generation = missing_value_generation

    def _set_model_missing_values(self, model_missing_values):
        warnings.warn(
            "Future versions of RDT will not support the 'model_missing_values' parameter. "
            "Please switch to using the 'missing_value_generation' parameter to select your "
            'strategy.',
            FutureWarning,
        )
        if model_missing_values is True:
            self._set_missing_value_generation('from_column')
        elif model_missing_values is False:
            self._set_missing_value_generation('random')

    @classmethod
    def get_name(cls):
        """Return transformer name.

        Returns:
            str:
                Transformer name.
        """
        return cls.__name__

    @classmethod
    def get_subclasses(cls):
        """Recursively find subclasses of this Baseline.

        Returns:
            list:
                List of all subclasses of this class.
        """
        subclasses = []
        for subclass in cls.__subclasses__():
            if abc.ABC not in subclass.__bases__:
                subclasses.append(subclass)

            subclasses += subclass.get_subclasses()

        return subclasses

    @classmethod
    def get_input_sdtype(cls):
        """Return the input sdtype supported by the transformer.

        Returns:
            string:
                Accepted input sdtype of the transformer.
        """
        warnings.warn(
            '`get_input_sdtype` is deprecated. Please use `get_supported_sdtypes` instead.',
            FutureWarning,
        )
        return cls.get_supported_sdtypes()[0]

    @classmethod
    def get_supported_sdtypes(cls):
        """Return the supported sdtypes by the transformer.

        Returns:
            list:
                Accepted input sdtypes of the transformer.
        """
        return cls.SUPPORTED_SDTYPES or [cls.INPUT_SDTYPE]

    def _get_output_to_property(self, property_):
        output = {}
        for output_column, properties in self.output_properties.items():
            # if 'sdtype' is not in the dict, ignore the column
            if property_ not in properties:
                continue
            if output_column is None:
                output[f'{self.column_prefix}'] = properties[property_]
            else:
                output[f'{self.column_prefix}.{output_column}'] = properties[property_]

        return output

    def get_output_sdtypes(self):
        """Return the output sdtypes produced by this transformer.

        Returns:
            dict:
                Mapping from the transformed column names to the produced sdtypes.
        """
        return self._get_output_to_property('sdtype')

    def get_next_transformers(self):
        """Return the suggested next transformer to be used for each column.

        Returns:
            dict:
                Mapping from transformed column names to the transformers to apply to each column.
        """
        return self._get_output_to_property('next_transformer')

    def is_generator(self):
        """Return whether this transformer generates new data or not.

        Returns:
            bool:
                Whether this transformer generates new data or not.
        """
        return bool(self.IS_GENERATOR)

    def get_input_column(self):
        """Return input column name for transformer.

        Returns:
            str:
                Input column name.
        """
        return self.columns[0]

    def get_output_columns(self):
        """Return list of column names created in ``transform``.

        Returns:
            list:
                Names of columns created during ``transform``.
        """
        return list(self._get_output_to_property('sdtype'))

    def _store_columns(self, columns, data):
        if isinstance(columns, tuple) and columns not in data:
            columns = list(columns)
        elif not isinstance(columns, list):
            columns = [columns]

        missing = set(columns) - set(data.columns)
        if missing:
            raise KeyError(f'Columns {missing} were not present in the data.')

        self.columns = columns

    @staticmethod
    def _get_columns_data(data, columns):
        if len(columns) == 1:
            columns = columns[0]

        return data[columns].copy()

    @staticmethod
    def _add_columns_to_data(data, transformed_data, transformed_names):
        """Add new columns to a ``pandas.DataFrame``.

        Args:
            - data (pd.DataFrame):
                The ``pandas.DataFrame`` to which the new columns have to be added.
            - transformed_data (pd.DataFrame, pd.Series, np.ndarray):
                The data of the new columns to be added.
            - transformed_names (list, np.ndarray):
                The names of the new columns to be added.

        Returns:
            ``pandas.DataFrame`` with the new columns added.
        """
        if transformed_names:
            if isinstance(transformed_data, (pd.Series, np.ndarray)):
                transformed_data = pd.DataFrame(transformed_data, columns=transformed_names)

            # When '#' is added to the column_prefix of a transformer
            # the columns of transformed_data and transformed_names don't match
            transformed_data.columns = transformed_names
            data = pd.concat([data, transformed_data.set_index(data.index)], axis=1)

        return data

    def _build_output_columns(self, data):
        self.column_prefix = '#'.join(self.columns)
        self.output_columns = self.get_output_columns()

        # make sure none of the generated `output_columns` exists in the data,
        # except when a column generates another with the same name
        output_columns = set(self.output_columns) - set(self.columns)
        repeated_columns = set(output_columns) & set(data.columns)
        while repeated_columns:
            warnings.warn(
                f'The output columns {repeated_columns} generated by the {self.get_name()} '
                'transformer already exist in the data (or they have already been generated '
                "by some other transformer). Appending a '#' to the column name to distinguish "
                'between them.'
            )
            self.column_prefix += '#'
            self.output_columns = self.get_output_columns()
            output_columns = set(self.output_columns) - set(self.columns)
            repeated_columns = set(output_columns) & set(data.columns)

    def __repr__(self):
        """Represent initialization of transformer as text.

        Returns:
            str:
                The name of the transformer followed by any non-default parameters.
        """
        class_name = self.__class__.get_name()
        custom_args = []
        args = inspect.getfullargspec(self.__init__)
        keys = args.args[1:]
        instanced = {
            key: getattr(self, key)
            for key in keys
            if key != 'model_missing_values' and hasattr(self, key)  # Remove after deprecation
        }

        defaults = args.defaults or []
        defaults = dict(zip(keys, defaults))
        if defaults == instanced:
            return f'{class_name}()'

        for arg, value in instanced.items():
            if arg not in defaults or defaults[arg] != value:
                custom_args.append(f'{arg}={repr(value)}')

        args_string = ', '.join(custom_args)
        return f'{class_name}({args_string})'

    def _fit(self, columns_data):
        """Fit the transformer to the data.

        Args:
            columns_data (pandas.DataFrame or pandas.Series):
                Data to transform.
        """
        raise NotImplementedError()

    def _set_seed(self, data):
        hash_value = self.columns[0]
        for _, row in data.head(5).iterrows():
            hash_value += str(row[self.columns[0]])

        hash_value = int(hashlib.sha256(hash_value.encode('utf-8')).hexdigest(), 16)
        self.random_seed = hash_value % ((2**32) - 1)  # maximum value for a seed
        self.random_states = {
            'fit': self.INITIAL_FIT_STATE,
            'transform': np.random.RandomState(self.random_seed),
            'reverse_transform': np.random.RandomState(self.random_seed + 1),
        }

    @random_state
    def fit(self, data, column):
        """Fit the transformer to a ``column`` of the ``data``.

        Args:
            data (pandas.DataFrame):
                The entire table.
            column (str):
                Column name. Must be present in the data.
        """
        self._store_columns(column, data)
        self._set_seed(data)
        columns_data = self._get_columns_data(data, self.columns)
        self._fit(columns_data)
        self._build_output_columns(data)

    def _transform(self, columns_data):
        """Transform the data.

        Args:
            columns_data (pandas.DataFrame or pandas.Series):
                Data to transform.

        Returns:
            pandas.DataFrame or pandas.Series:
                Transformed data.
        """
        raise NotImplementedError()

    @random_state
    def transform(self, data):
        """Transform the `self.columns` of the `data`.

        Args:
            data (pandas.DataFrame):
                The entire table.

        Returns:
            pd.DataFrame:
                The entire table, containing the transformed data.
        """
        # if `data` doesn't have the columns that were fitted on, don't transform
        if any(column not in data.columns for column in self.columns):
            return data

        data = data.copy()
        columns_data = self._get_columns_data(data, self.columns)
        transformed_data = self._transform(columns_data)
        data = data.drop(self.columns, axis=1)
        data = self._add_columns_to_data(data, transformed_data, self.output_columns)

        return data

    def fit_transform(self, data, column):
        """Fit the transformer to a `column` of the `data` and then transform it.

        Args:
            data (pandas.DataFrame):
                The entire table.
            column (str):
                A column name.

        Returns:
            pd.DataFrame:
                The entire table, containing the transformed data.
        """
        self.fit(data, column)
        return self.transform(data)

    def _reverse_transform(self, columns_data):
        """Revert the transformations to the original values.

        Args:
            columns_data (pandas.DataFrame or pandas.Series):
                Data to revert.

        Returns:
            pandas.DataFrame or pandas.Series:
                Reverted data.
        """
        raise NotImplementedError()

    @random_state
    def reverse_transform(self, data):
        """Revert the transformations to the original values.

        Args:
            data (pandas.DataFrame):
                The entire table.

        Returns:
            pandas.DataFrame:
                The entire table, containing the reverted data.
        """
        # if `data` doesn't have the columns that were transformed, don't reverse_transform
        if any(column not in data.columns for column in self.output_columns):
            return data

        data = data.copy()
        columns_data = self._get_columns_data(data, self.output_columns)
        original_missing_values = self.missing_value_generation
        if self.missing_value_generation is not None and pd.isna(columns_data).any().any():
            warnings.warn(
                "The 'missing_value_generation' parameter is set to '"
                f"{self.missing_value_generation}' but the data already contains missing values."
                ' Missing value generation will be skipped.',
                UserWarning,
            )
            self.missing_value_generation = None

        reversed_data = self._reverse_transform(columns_data)
        self.missing_value_generation = original_missing_values
        data = data.drop(self.output_columns, axis=1)
        data = self._add_columns_to_data(data, reversed_data, self.columns)

        return data




class BaseMultiColumnTransformer(BaseTransformer):
    """Base class for all multi column transformers.

    The ``BaseMultiColumnTransformer`` class contains methods that must be implemented
    in order to create a new multi column transformer.

    Attributes:
        columns_to_sdtypes (dict):
            Dictionary mapping each column to its sdtype.
    """

    def __init__(self):
        super().__init__()
        self.columns_to_sdtypes = {}

    def get_input_column(self):
        """Override ``get_input_column`` method from ``BaseTransformer``.

        Raise an error because for multi column transformers, ``get_input_columns``
        must be used instead.
        """
        raise NotImplementedError(
            'MultiColumnTransformers does not have a single input column.'
            'Please use ``get_input_columns`` instead.'
        )

    def get_input_columns(self):
        """Return input column name for transformer.

        Returns:
            list:
                Input column names.
        """
        return self.columns

    def _get_prefix(self):
        """Return the prefix of the output columns.

        Returns:
            str:
                Prefix of the output columns.
        """
        raise NotImplementedError()

    def _get_output_to_property(self, property_):
        self.column_prefix = self._get_prefix()
        output = {}
        for output_column, properties in self.output_properties.items():
            # if 'sdtype' is not in the dict, ignore the column
            if property_ not in properties:
                continue

            if self.column_prefix is None:
                output[f'{output_column}'] = properties[property_]
            else:
                output[f'{self.column_prefix}.{output_column}'] = properties[property_]

        return output

    def _validate_columns_to_sdtypes(self, data, columns_to_sdtypes):
        """Check that all the columns in ``columns_to_sdtypes`` are present in the data."""
        missing = set(columns_to_sdtypes.keys()) - set(data.columns)
        if missing:
            missing_to_print = ', '.join(missing)
            raise ValueError(f'Columns ({missing_to_print}) are not present in the data.')

    @classmethod
    def _validate_sdtypes(cls, columns_to_sdtypes):
        raise NotImplementedError()

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.DataFrame):
                Data to transform.
        """
        raise NotImplementedError()

    @random_state
    def fit(self, data, columns_to_sdtypes):
        """Fit the transformer to a ``column`` of the ``data``.

        Args:
            data (pandas.DataFrame):
                The entire table.
            columns_to_sdtypes (dict):
                Dictionary mapping each column to its sdtype.
        """
        self._validate_columns_to_sdtypes(data, columns_to_sdtypes)
        self.columns_to_sdtypes = columns_to_sdtypes
        self._store_columns(list(self.columns_to_sdtypes.keys()), data)
        self._set_seed(data)
        columns_data = self._get_columns_data(data, self.columns)
        self._fit(columns_data)
        self._build_output_columns(data)

    def fit_transform(self, data, columns_to_sdtypes):
        """Fit the transformer to a `column` of the `data` and then transform it.

        Args:
            data (pandas.DataFrame):
                The entire table.
            columns_to_sdtypes (dict):
                Dictionary mapping each column to its sdtype.

        Returns:
            pd.DataFrame:
                The entire table, containing the transformed data.
        """
        self.fit(data, columns_to_sdtypes)
        return self.transform(data)


"""Transformer for data that contains Null values."""

class NullTransformer:

    nulls = None
    _missing_value_generation = None
    _missing_value_replacement = None
    _null_percentage = None

    def __init__(self, missing_value_replacement=None, missing_value_generation='random'):
        self._missing_value_replacement = missing_value_replacement
        if missing_value_generation not in (None, 'from_column', 'random'):
            raise ValueError(
                "'missing_value_generation' must be one of the following values: "
                "None, 'from_column' or 'random'."
            )

        self._missing_value_generation = missing_value_generation
        self._min_value = None
        self._max_value = None

    def models_missing_values(self):
        """Indicate whether this transformer creates a null column on transform.

        Returns:
            bool:
                Whether a null column is created on transform.
        """
        return self._missing_value_generation == 'from_column'

    def _get_missing_value_replacement(self, data):

        if self._missing_value_replacement is None:
            return None

        if self._missing_value_replacement in {'mean', 'mode', 'random'} and pd.isna(data).all():
            msg = (
                f"'missing_value_replacement' cannot be set to '{self._missing_value_replacement}'"
                ' when the provided data only contains NaNs. Using 0 instead.'
            )
            LOGGER.info(msg)
            return 0

        if self._missing_value_replacement == 'mean':
            return data.mean()

        if self._missing_value_replacement == 'mode':
            return data.mode(dropna=True)[0]

        return self._missing_value_replacement

    def fit(self, data):
        """Fit the transformer to the data.

        Evaluate if the transformer has to create the null column or not.

        Args:
            data (pandas.Series):
                Data to transform.
        """
        self._missing_value_replacement = self._get_missing_value_replacement(data)
        if self._missing_value_replacement == 'random':
            self._min_value = data.min()
            self._max_value = data.max()

        if self._missing_value_generation is not None:
            null_values = data.isna().to_numpy()
            self.nulls = null_values.any()

            if not self.nulls and self.models_missing_values():
                self._missing_value_generation = None
                guidance_message = (
                    f'Guidance: There are no missing values in column {data.name}. '
                    'Extra column not created.'
                )
                LOGGER.info(guidance_message)

            if self._missing_value_generation == 'random':
                self._null_percentage = null_values.sum() / len(data)

    def _set_fitted_parameters(self, null_ratio):
        """Manually set the parameters on the transformer to get it into a fitted state.

        Args:
            null_ratio (float):
                The fraction of values to replace with null values.
        """
        if null_ratio < 0 or null_ratio > 1.0:
            raise ValueError('null_ratio should be a value between 0 and 1.')

        if null_ratio != 0:
            self.nulls = True
            self._null_percentage = null_ratio

    def transform(self, data):
        """Replace null values with the indicated ``missing_value_replacement``.

        If required, create the null indicator column.

        Args:
            data (pandas.Series or numpy.ndarray):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        isna = data.isna()
        if self._missing_value_replacement == 'random':
            data_mask = list(
                np.random.uniform(low=self._min_value, high=self._max_value, size=len(data))
            )
            data = data.mask(data.isna(), data_mask)

        elif isna.any() and self._missing_value_replacement is not None:
            data = data.infer_objects().fillna(self._missing_value_replacement)

        if self._missing_value_generation == 'from_column':
            return pd.concat([data, isna.astype(np.float64)], axis=1).to_numpy()

        return data.to_numpy()

    def reverse_transform(self, data):

        data = data.copy()
        if self._missing_value_generation == 'from_column':
            if self.nulls:
                isna = data[:, 1] > 0.5

            data = data[:, 0]

        elif self.nulls:
            isna = np.random.random((len(data),)) < self._null_percentage

        data = pd.Series(data)

        if self.nulls and isna.any():
            data.loc[isna] = np.nan

        return data
