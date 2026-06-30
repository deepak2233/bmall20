# Install Prediction Model (IPM) — Training & Serving Pipeline

An end-to-end pipeline that trains a binary classifier to predict the probability
that an ad impression leads to an app install, logs quality metrics to MLflow,
and packages the **whole preprocessing + model graph into a single ONNX file**
for low-latency real-time bidding.

```
raw impressions ─▶ load + temporal split ─▶ featurize ─▶ fit LightGBM (early stop)
                                                              │
                          MLflow ◀── metrics/params/artifacts │
                                                              ▼
                                          export ONNX (preproc + model in one graph)
                                                              │
                                                  ┌───────────┴───────────┐
                                          parity check            FastAPI /predict
```

## Quickstart

```bash
pip install -r requirements.txt            # or: make install

# The real dataset.parquet is private and NOT in this repo. Generate a
# schema-faithful synthetic stand-in so everything runs out of the box:
python scripts/generate_synthetic_data.py --rows 60000 --out data/sample.parquet

make eda      # reproduce the data analysis that drove the design
make train    # train -> evaluate -> export ONNX -> verify parity -> log to MLflow
make test     # unit + integration tests (incl. sklearn↔ONNX parity)
make serve    # REST inference on http://localhost:8080  (extra credit)
```

To train on the **real** data, point `config/config.yaml` → `data.path` at
`dataset.parquet` (parquet directory or file) and run `make train`. The loader
auto-detects parquet vs csv.

## Repository layout

```
config/config.yaml          # all knobs (data split, model HPs, tracking)
src/ipm/
  schema.py                 # single source of truth for column names/types
  config.py                 # typed config dataclasses
  data.py                   # robust loading + temporal train/valid/test split
  featurizer.py             # STATELESS feature engineering (train == serve)
  model.py                  # sklearn pipeline build + ONNX export + ONNX scoring
  metrics.py                # logloss, ROC-AUC, PR-AUC, Brier
  train.py                  # orchestration + MLflow logging
  serve.py                  # FastAPI low-latency inference service
scripts/
  generate_synthetic_data.py# schema-faithful data generator (real data is private)
  eda.py                    # reproducible data profiling
  example_request.py        # sample /predict call
tests/                      # featurizer contract + end-to-end + ONNX parity
docs/DESIGN.md              # 2-page architecture write-up
Dockerfile                  # slim onnxruntime-only serving image
```

## What the model produces

A calibrated probability `P(install)` per impression. After training, `artifacts/`
contains:

| file | purpose |
|------|---------|
| `model.onnx` | preprocessing **and** GBDT in one graph; per-feature typed inputs |
| `featurizer_config.json` | versioned feature contract for the serving layer |
| `metadata.json` | metrics, best iteration, feature lists, opset, train size |

## Design highlights

- **One ONNX graph for the full pipeline** — imputation, scaling and one-hot
  vocabularies are baked in, so there is no learned preprocessing to drift
  between training and serving. Verified by an automatic parity check
  (`|sklearn − onnx| < 1e-4`, typically ~1e-7).
- **Stateless featurizer** — the only Python that runs at serve time is fixed
  arithmetic/string ops, guaranteeing identical features in training and serving.
- **Temporal split** — train on the past, evaluate on the future, matching
  production and preventing look-ahead leakage.
- **Calibrated probabilities by default** — a bidding system bids on the
  probability itself, so we train on the natural class distribution rather than
  reweighting (which improves nothing here and degrades calibration). See
  `docs/DESIGN.md`.
- **Graceful unknowns** — unseen `appid`/`country` and missing fields map to safe
  defaults instead of crashing (covered by tests).

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full architecture, the data
takeaways, and what a production system would add.

> The provided dataset is private (per the assignment) and is intentionally not
> committed. `data/` is git-ignored.
