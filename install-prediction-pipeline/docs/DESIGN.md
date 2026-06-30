# Install Prediction Model — Design Write-Up

## 1. Data analysis → takeaways that shaped the design

Profiling the provided 10k anonymised sample (`scripts/eda.py`) surfaced the
properties that every downstream decision is built on:

| Observation (sample) | Implication | What the pipeline does |
|---|---|---|
| **Install rate ≈ 3.25%** (heavy imbalance) | ROC-AUC alone is misleading; need ranking + calibration metrics | Report log-loss, ROC-AUC, **PR-AUC**, **Brier**; train calibrated (see §4) |
| **Missingness is informative** — `count_user_clicks_7` null → 1.5% install vs 3.5% overall; `user_install_profile` 22% null, `memory_total` 2% null | Imputing silently destroys signal | Emit explicit `*_missing` flags, then impute |
| **High-cardinality IDs** — `user_id` ≈ unique (9939/10000), `appid` 222, `sdkappid` 257, `country` 144 | `user_id` can't generalise (leakage/overfit); others need unseen-category handling | Drop `user_id` as a feature; one-hot with `handle_unknown="ignore"` |
| **`memory_total` in bytes** (1.8–17 GB); counts heavy-tailed (p99≈30, max in hundreds) | Raw scales hurt tree splits / linear terms | Rescale memory→GB, `log1p` the counts |
| **`user_install_profile`** = space-delimited app list, median 18 apps | Sequence can't enter ONNX directly | Summarise as length + presence flag (embeddings = future work) |
| **17-day timestamp window** | Random split would leak the future | **Temporal** train/valid/test split; timestamp used only as cyclical hour/day-of-week |
| **Data-quality quirk** — clicks > impressions in ~1% of rows; 60 users with >1 impression | Don't assume clean joins; avoid per-user leakage | Robust coercion (`NaN` for bad values); time-ordered split keeps a user's later impressions out of train |

The headline engineering point: **signal is weak and messy**, so the value is in
a correct, reproducible, serving-faithful pipeline rather than model tuning —
exactly the assignment's framing.

## 2. Architecture

```
                 ┌──────────────────────── TRAINING ────────────────────────┐
 dataset.parquet │  data.load_raw ──▶ temporal_split ──▶ featurizer.transform │
   (or .csv)     │        (column projection)                │                │
                 │                                           ▼                │
                 │                     sklearn Pipeline:  ColumnTransformer    │
                 │                       • numeric: median-impute → scale      │
                 │                       • categorical: one-hot (ignore unk)   │
                 │                                  └─▶ LightGBM (early stop)  │
                 │                                           │                 │
                 │   MLflow ◀── params · metrics · artifacts │                │
                 │                                           ▼                │
                 │            model.export_onnx  →  model.onnx  (ONE graph)    │
                 │                                           │                 │
                 │            parity check: |sklearn − onnx| < 1e-4            │
                 └───────────────────────────────────────────┼───────────────┘
                                                              ▼
                 ┌──────────────────────── SERVING ─────────────────────────┐
   JSON request  │  featurizer.transform (same code) ──▶ ONNX Runtime session │  P(install)
   ───────────▶  │  (stateless Python: parse/derive)     (CPU, 1 thread)      │ ──────────▶
                 └────────────────────────────────────────────────────────────┘
```

**Two-stage preprocessing, on purpose.** ONNX's `ai.onnx.ml` operators cannot
split strings, parse timestamps or count sequence tokens. So feature engineering
is split:

1. **Stateless featurizer** (`featurizer.py`) — fixed arithmetic/string ops with
   *no learned parameters*. Identical code runs in training and serving, so there
   is zero feature skew by construction.
2. **Learned preprocessing + model** (imputer medians, scaler stats, one-hot
   vocabularies, the GBDT) — fitted by sklearn and exported **whole** into a
   single `model.onnx`. The serving runtime loads one artifact.

## 3. Why these technology choices

- **LightGBM** — strong on tabular/heavy-tailed/missing data with little tuning,
  trains in seconds, and its trees compile to fast ONNX ops. Framework was free
  to choose; this maximises engineering-quality-per-hour.
- **ONNX + ONNX Runtime** — language-agnostic, no Python/sklearn/LightGBM at
  serve time (the Docker image is `onnxruntime`-only), single-row latency
  ~0.02 ms in-process. The whole pipeline in one graph eliminates a class of
  train/serve bugs.
- **MLflow** — logs hyper-parameters, all metrics, and the ONNX + config +
  metadata artifacts per run, giving reproducibility and run comparison.
- **FastAPI** — minimal, typed REST surface for the extra-credit service.

## 4. Edge cases handled

- **Class imbalance & calibration.** A bidding system bids on `P(install)`, so
  calibration matters. Empirically (`balance_classes`), reweighting with
  `scale_pos_weight` left ROC-AUC/PR-AUC unchanged or worse **and** inflated
  probabilities (mean prediction drifted above the base rate). Default is
  therefore **unweighted training on the natural distribution**, which is
  well-calibrated (mean prediction ≈ base rate; Brier improves). Reweighting
  remains a one-line config option for recall-oriented variants.
- **Unseen categories** at serving (new `appid`, `country`) → one-hot
  `handle_unknown="ignore"` yields an all-zero block instead of an error (tested).
- **Missing / partial requests** → featurizer fills numeric `NaN` (imputed in
  graph) and a categorical sentinel; a request with only a couple of fields still
  scores (tested).
- **Bad/dirty values** → numeric coercion to `NaN`, rows without a usable
  timestamp dropped, target accepted as bool / `"True"/"False"` / 0-1.
- **Look-ahead leakage** → temporal split; timestamp never used as raw magnitude.
- **Export correctness** → training fails loudly if sklearn↔ONNX probabilities
  diverge by >1e-4.

## 5. What a production system would add

- **Data & features:** a feature store (e.g. Feast) so the 7-day counts and
  install profile are computed once and shared online/offline, killing
  train/serve skew at the source; point-in-time correct joins; a richer
  `user_install_profile` representation (hashed bag-of-apps or a learned
  embedding) instead of just its length.
- **Scale:** the real file is 1.5 GB — swap eager pandas for chunked/streaming
  reads (PyArrow datasets, Polars, or Spark/Ray) and out-of-core or distributed
  training; partition parquet by date for cheap temporal slicing.
- **Orchestration:** a scheduler (Airflow/Argo/Kubeflow) running daily retrains
  with data-freshness and schema checks (Great Expectations) as gates.
- **Model management:** MLflow Model Registry with staging→prod promotion, a
  champion/challenger gate on hold-out PR-AUC + calibration, and automatic
  rollback.
- **Serving:** containerised ONNX Runtime behind the bidder with autoscaling,
  p99-latency SLOs, request/response logging, and shadow/canary deploys; batch
  the bid candidates per request to amortise overhead.
- **Monitoring:** online vs offline metric parity, feature drift (PSI/KL) and
  prediction-distribution drift, calibration tracking against realised installs
  (postbacks arrive delayed), and alerting that triggers retraining.
- **Governance:** reproducible runs (data + code + config hashes), feature/data
  lineage, and privacy handling for `user_id` / install history.
