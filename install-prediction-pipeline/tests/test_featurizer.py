"""Unit tests for the stateless featurizer — the train/serve contract."""
import numpy as np
import pandas as pd
import pytest

from ipm import featurizer, schema


def _raw_row(**overrides):
    base = {
        "user_id": "u1",
        "country": "US",
        "device_os": "Android",
        "count_user_impressions_7": 10,
        "appid": "app.a",
        "sdkappid": "sdk.b",
        "memory_total": 4_000_000_000,
        "count_user_clicks_7": 3,
        "session_count_7d": 5,
        "user_install_profile": "a.b c.d e.f",
        "postback": 0,
        "timestamp": 1_774_000_000_000,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_output_columns_and_order():
    feats = featurizer.transform(_raw_row())
    assert list(feats.columns) == featurizer.ALL_FEATURES


def test_numeric_transforms():
    feats = featurizer.transform(_raw_row())
    row = feats.iloc[0]
    assert row["memory_gb"] == pytest.approx(4.0, abs=1e-6)
    assert row["log_impressions_7"] == pytest.approx(np.log1p(10))
    assert row["ctr_7"] == pytest.approx(3 / (10 + 1))
    assert row["profile_len"] == 3.0


def test_missing_flags_are_informative():
    feats = featurizer.transform(
        _raw_row(count_user_clicks_7=None, user_install_profile=None, memory_total=None)
    )
    row = feats.iloc[0]
    assert row["clicks_missing"] == 1.0
    assert row["profile_missing"] == 1.0
    assert row["memory_missing"] == 1.0
    assert row["profile_len"] == 0.0  # null profile -> length 0


def test_categorical_missing_sentinel():
    feats = featurizer.transform(_raw_row(country=None))
    assert feats.iloc[0][schema.COUNTRY] == schema.CATEGORICAL_MISSING


def test_cyclical_time_bounds():
    feats = featurizer.transform(_raw_row())
    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
        assert -1.0 <= feats.iloc[0][col] <= 1.0


def test_handles_completely_empty_optional_fields():
    # A request with only the bare minimum should still featurize without error.
    feats = featurizer.transform(_raw_row(
        country=None, device_os=None, appid=None, sdkappid=None,
        count_user_clicks_7=None, session_count_7d=None,
        memory_total=None, user_install_profile=None,
    ))
    assert feats.shape == (1, len(featurizer.ALL_FEATURES))
    assert not feats[featurizer.CATEGORICAL_FEATURES].isnull().any().any()


def test_onnx_input_split_shapes_and_dtypes():
    feats = featurizer.transform(pd.concat([_raw_row(), _raw_row()], ignore_index=True))
    feed = featurizer.split_onnx_inputs(feats)
    assert set(feed) == set(featurizer.ALL_FEATURES)
    for name in featurizer.NUMERIC_FEATURES:
        assert feed[name].shape == (2, 1)
        assert feed[name].dtype == np.float32
    for name in featurizer.CATEGORICAL_FEATURES:
        assert feed[name].shape == (2, 1)
        assert feed[name].dtype == object
