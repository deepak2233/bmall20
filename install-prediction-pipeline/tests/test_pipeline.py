"""Integration tests: data loading, temporal split, fit, and ONNX parity.

These exercise the full path end-to-end on a small synthetic frame so the suite
runs in seconds while still proving the export is correct.
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_synthetic_data import generate  # noqa: E402

from ipm import data, featurizer, model, schema  # noqa: E402
from ipm.config import ModelConfig  # noqa: E402


@pytest.fixture(scope="module")
def frame():
    return data._coerce_types(generate(4000, seed=1))


def test_temporal_split_is_ordered_and_disjoint(frame):
    frame = frame.sort_values(schema.TIMESTAMP).reset_index(drop=True)
    train, valid, test = data.temporal_split(frame, 0.15, 0.15)
    assert len(train) + len(valid) + len(test) == len(frame)
    # Train happens strictly before test in time (no look-ahead leakage).
    assert train[schema.TIMESTAMP].max() <= test[schema.TIMESTAMP].min()


def test_split_rejects_invalid_fractions(frame):
    with pytest.raises(ValueError):
        data.temporal_split(frame, 0.6, 0.6)


def test_fit_and_onnx_parity(tmp_path, frame):
    frame = frame.sort_values(schema.TIMESTAMP).reset_index(drop=True)
    train, _, test = data.temporal_split(frame, 0.15, 0.15)
    X_train = featurizer.transform(train)
    y_train = train[schema.TARGET].to_numpy()
    X_test = featurizer.transform(test)

    spw = model.compute_scale_pos_weight(y_train)
    pipe = model.build_pipeline(ModelConfig(n_estimators=60), scale_pos_weight=spw)
    pipe.fit(X_train, y_train)

    onnx_path = model.export_onnx(pipe, opset=17, output_path=tmp_path / "m.onnx")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    sk = pipe.predict_proba(X_test)[:, 1]
    on = model.predict_proba_onnx(sess, X_test)
    assert np.abs(sk - on).max() < 1e-4


def test_unseen_category_does_not_crash(tmp_path, frame):
    frame = frame.sort_values(schema.TIMESTAMP).reset_index(drop=True)
    train, _, test = data.temporal_split(frame, 0.15, 0.15)
    X_train = featurizer.transform(train)
    y_train = train[schema.TARGET].to_numpy()
    pipe = model.build_pipeline(ModelConfig(n_estimators=30), scale_pos_weight=1.0)
    pipe.fit(X_train, y_train)
    onnx_path = model.export_onnx(pipe, opset=17, output_path=tmp_path / "m.onnx")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # A brand-new appid/country never seen in training must still score.
    novel = test.iloc[:5].copy()
    novel[schema.APPID] = "totally.new.app"
    novel[schema.COUNTRY] = "ZZ"
    feats = featurizer.transform(novel)
    probs = model.predict_proba_onnx(sess, feats)
    assert probs.shape == (5,)
    assert np.all((probs >= 0) & (probs <= 1))
