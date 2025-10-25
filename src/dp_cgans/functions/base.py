"""Base Constraint class."""

import copy
import importlib
import inspect
import logging
import warnings
import pandas as pd

from dp_cgans.functions.gaussian import GaussianUnivariate, GaussianMultivariate
from dp_cgans.Transformers.hyper_transformer import HyperTransformer


LOGGER = logging.getLogger(__name__)


def _get_qualified_name(obj):
    """Return the Fully Qualified Name from an instance or class."""
    module = obj.__module__
    if hasattr(obj, '__name__'):
        obj_name = obj.__name__
    else:
        obj_name = obj.__class__.__name__

    return module + '.' + obj_name


def _module_contains_callable_name(obj):
    """Return if module contains the name of the callable object."""
    if hasattr(obj, '__name__'):
        obj_name = obj.__name__
    else:
        obj_name = obj.__class__.__name__
    return obj_name in importlib.import_module(obj.__module__).__dict__


def get_subclasses(cls):
    """Recursively find subclasses for the current class object."""
    subclasses = dict()
    for subclass in cls.__subclasses__():
        subclasses[subclass.__name__] = subclass
        subclasses.update(get_subclasses(subclass))

    return subclasses


def import_object(obj):
    """Import an object from its qualified name."""
    if isinstance(obj, str):
        package, name = obj.rsplit('.', 1)
        return getattr(importlib.import_module(package), name)

    return obj


class ConstraintMeta(type):
    """Metaclass for Constraints.

    This metaclass replaces the ``__init__`` method with a new function
    that stores the arguments passed to the __init__ method in a dict
    as the attribute ``__kwargs__``.

    This allows us to later on dump the class definition as a dict.
    """

    def __init__(self, name, bases, attr):
        super().__init__(name, bases, attr)

        old__init__ = self.__init__
        signature = inspect.signature(old__init__)
        arg_names = list(signature.parameters.keys())[1:]

        def __init__(self, *args, **kwargs):
            class_name = self.__class__.__name__
            if name == class_name:
                self.__kwargs__ = copy.deepcopy(kwargs)
                self.__kwargs__.update(dict(zip(arg_names, args)))

            old__init__(self, *args, **kwargs)

        __init__.__doc__ = old__init__.__doc__
        __init__.__signature__ = signature
        self.__init__ = __init__


class Constraint(metaclass=ConstraintMeta):

    constraint_columns = ()
    rebuild_columns = ()
    _hyper_transformer = None
    _columns_model = None

    def _identity(self, table_data):
        return table_data

    def __init__(self, handling_strategy, fit_columns_model=False):
        self.fit_columns_model = fit_columns_model
        if handling_strategy == 'transform':
            self.filter_valid = self._identity
        elif handling_strategy == 'reject_sampling':
            self.rebuild_columns = ()
            self.transform = self._identity
            self.reverse_transform = self._identity
        elif handling_strategy != 'all':
            raise ValueError('Unknown handling strategy: {}'.format(handling_strategy))

    def _fit(self, table_data):
        del table_data

    def fit(self, table_data):
        self._fit(table_data)

        if self.fit_columns_model and len(self.constraint_columns) > 1:
            data_to_model = table_data[list(self.constraint_columns)]
            self._hyper_transformer = HyperTransformer(default_data_type_transformers={
                'categorical': 'OneHotEncodingTransformer',
            })
            transformed_data = self._hyper_transformer.fit_transform(data_to_model)
            self._columns_model = GaussianMultivariate(
                distribution=GaussianUnivariate
            )
            self._columns_model.fit(transformed_data)

    def _transform(self, table_data):
        return table_data

    def _reject_sample(self, num_rows, conditions):
        sampled = self._columns_model.sample(
            num_rows=num_rows,
            conditions=conditions
        )
        sampled = self._hyper_transformer.reverse_transform(sampled)
        valid_rows = sampled[self.is_valid(sampled)]
        counter = 0
        total_sampled = num_rows

        while len(valid_rows) < num_rows:
            num_valid = len(valid_rows)
            if counter >= 100:
                if len(valid_rows) == 0:
                    error = 'Could not get enough valid rows within 100 trials.'
                    raise ValueError(error)
                else:
                    multiplier = num_rows // num_valid
                    num_rows_missing = num_rows % num_valid
                    remainder_rows = valid_rows.iloc[0:num_rows_missing, :]
                    valid_rows = pd.concat([valid_rows] * multiplier + [remainder_rows],
                                           ignore_index=True)
                    break

            remaining = num_rows - num_valid
            valid_probability = (num_valid + 1) / (total_sampled + 1)
            max_rows = num_rows * 10
            num_to_sample = min(int(remaining / valid_probability), max_rows)
            total_sampled += num_to_sample
            new_sampled = self._columns_model.sample(
                num_rows=num_to_sample,
                conditions=conditions
            )
            new_sampled = self._hyper_transformer.reverse_transform(new_sampled)
            new_valid_rows = new_sampled[self.is_valid(new_sampled)]
            valid_rows = pd.concat([valid_rows, new_valid_rows], ignore_index=True)
            counter += 1

        return valid_rows.iloc[0:num_rows, :]

    def _sample_constraint_columns(self, table_data):
        condition_columns = [c for c in self.constraint_columns if c in table_data.columns]
        grouped_conditions = table_data[condition_columns].groupby(condition_columns)
        all_sampled_rows = list()
        for group, df in grouped_conditions:
            if not isinstance(group, tuple):
                group = [group]

            transformed_condition = self._hyper_transformer.transform(df).iloc[0].to_dict()
            sampled_rows = self._reject_sample(
                num_rows=df.shape[0],
                conditions=transformed_condition
            )
            all_sampled_rows.append(sampled_rows)

        sampled_data = pd.concat(all_sampled_rows, ignore_index=True)
        return sampled_data

    def _validate_constraint_columns(self, table_data):
        missing_columns = [col for col in self.constraint_columns if col not in table_data.columns]
        if missing_columns:
            if not self._columns_model:
                warning_message = (
                    'When `fit_columns_model` is False and we are conditioning on a subset '
                    'of the constraint columns, conditional sampling uses reject sampling '
                    'which can be slow. Changing `fit_columns_model` to True can improve '
                    'the performance.'
                )
                warnings.warn(warning_message, UserWarning)

            all_columns_missing = len(missing_columns) == len(self.constraint_columns)
            if self._columns_model is None or all_columns_missing:
                raise ValueError()

            else:
                sampled_data = self._sample_constraint_columns(table_data)
                other_columns = [c for c in table_data.columns if c not in self.constraint_columns]
                sampled_data[other_columns] = table_data[other_columns]
                return sampled_data

        return table_data

    def transform(self, table_data):
        table_data = self._validate_constraint_columns(table_data)
        return self._transform(table_data)

    def fit_transform(self, table_data):
        self.fit(table_data)
        return self.transform(table_data)

    def reverse_transform(self, table_data):
        return table_data

    def is_valid(self, table_data):
        return pd.Series(True, index=table_data.index)

    def filter_valid(self, table_data):
        valid = self.is_valid(table_data)
        invalid = sum(~valid)
        if invalid:
            LOGGER.debug('%s: %s invalid rows out of %s.',
                         self.__class__.__name__, sum(~valid), len(valid))

        if isinstance(valid, pd.Series):
            return table_data[valid.values]

        return table_data[valid]

    @classmethod
    def from_dict(cls, constraint_dict):
        constraint_dict = constraint_dict.copy()
        constraint_class = constraint_dict.pop('constraint')
        subclasses = get_subclasses(cls)
        if isinstance(constraint_class, str):
            if '.' in constraint_class:
                constraint_class = import_object(constraint_class)
            else:
                constraint_class = subclasses[constraint_class]

        return constraint_class(**constraint_dict)

    def to_dict(self):
        constraint_dict = {
            'constraint': _get_qualified_name(self.__class__),
        }

        for key, obj in copy.deepcopy(self.__kwargs__).items():
            if callable(obj) and _module_contains_callable_name(obj):
                constraint_dict[key] = _get_qualified_name(obj)
            else:
                constraint_dict[key] = obj

        return constraint_dict
