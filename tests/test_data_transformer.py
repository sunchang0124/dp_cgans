import numpy as np
import pandas as pd
import pytest

from dp_cgans.synthesizers.data_transformer import DataTransformer


def _decimals(series):
    """Count decimal places actually present in a numeric pandas Series."""
    return series.astype(str).str.extract(r"\.(\d+)$")[0].str.rstrip("0").str.len().fillna(0).max()


@pytest.mark.parametrize(
    "values, expected",
    [
        ([1, 2, 3], 0),
        ([1.0, 2.0, 3.0], 0),
        ([1.5, 2.25, 3.0], 2),
        ([0.1, 0.123, 7.0], 3),
        ([np.nan, 1.25, 2.5], 2),
        ([np.nan, np.nan], None),
        ([1 / 3, 2 / 3], None),  # needs > 15 decimals, leave untouched
    ],
)
def test_learn_decimals(values, expected):
    assert DataTransformer._learn_decimals(np.array(values, dtype=float)) == expected


def test_inverse_transform_keeps_real_data_precision():
    rng = np.random.default_rng(0)
    n = 300
    real = pd.DataFrame({
        "two_dec": np.round(rng.normal(50, 10, n), 2),
        "one_dec": np.round(rng.uniform(0, 100, n), 1),
        "integer": rng.integers(18, 90, n),
        "category": rng.choice(["a", "b", "c"], n),
    })

    transformer = DataTransformer()
    transformer.fit(real, discrete_columns=["category"])
    assert transformer._column_decimals == {"two_dec": 2, "one_dec": 1, "integer": 0}

    # Perturb the encoded matrix the way a generator would, then decode.
    encoded = transformer.transform(real)
    noisy = encoded + rng.normal(0, 0.05, encoded.shape)
    synthetic = transformer.inverse_transform(noisy)

    assert _decimals(synthetic["two_dec"]) <= 2
    assert _decimals(synthetic["one_dec"]) <= 1
    assert synthetic["integer"].dtype == real["integer"].dtype
    assert list(synthetic.columns) == list(real.columns)


def test_inverse_transform_without_learned_decimals_still_works():
    """Models pickled before rounding existed have no ``_column_decimals``."""
    real = pd.DataFrame({"x": np.linspace(0, 1, 50)})
    transformer = DataTransformer()
    transformer.fit(real)
    del transformer._column_decimals
    out = transformer.inverse_transform(transformer.transform(real))
    assert out.shape == real.shape
