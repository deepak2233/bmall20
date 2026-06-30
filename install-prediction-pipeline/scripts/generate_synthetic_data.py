"""Generate a synthetic dataset matching the IPM schema.

The real ``dataset.parquet`` (1.5 GB) and the anonymised sample are private and
are intentionally **not** committed to this repository. This generator produces
a schema-faithful stand-in so the pipeline is runnable end-to-end by anyone who
clones the repo, and so the unit tests have data to run against.

It reproduces the statistical quirks observed during the data analysis:
  * ~3% install base-rate (heavy imbalance)
  * informative missingness in clicks / memory / sessions / install profile
  * memory_total in bytes, heavy-tailed counts, space-delimited app lists
  * a ~17-day timestamp window

Usage:  python scripts/generate_synthetic_data.py --rows 50000 --out data/sample.parquet
"""
from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

COUNTRIES = ["US", "BR", "ID", "TR", "MX", "IN", "GB", "DE", "FR", "JP", "NG", "PH"]
OSES = ["Android", "iOS"]


def _app_tokens(n: int, rng: np.random.Generator) -> list[str]:
    words = ["jade", "uniform", "spark", "haven", "silk", "fern", "drift", "bold",
             "vault", "shore", "warm", "aspen", "port", "romeo", "bloom", "mesa",
             "sage", "tango", "calm", "vast", "pulse", "quebec", "rapid", "canyon"]
    return [".".join(rng.choice(words, size=rng.integers(2, 4))) for _ in range(n)]


def generate(rows: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    app_pool = _app_tokens(250, rng)
    sdk_pool = _app_tokens(260, rng)

    impressions = rng.poisson(3.5, rows)
    clicks = np.minimum(impressions, rng.poisson(2.0, rows)).astype(float)
    sessions = rng.poisson(7.0, rows).astype(float)
    memory = rng.uniform(1.8e9, 1.7e10, rows)

    # Base time window ~17 days starting at a fixed epoch (ms).
    t0 = 1_774_000_000_000
    timestamp = t0 + rng.integers(0, 17 * 24 * 3600 * 1000, rows)

    profile_len = rng.integers(0, 80, rows)
    install_profile = [
        " ".join(rng.choice(app_pool, size=int(k))) if k > 0 else ""
        for k in profile_len
    ]

    # A learnable-but-noisy logit: real signal from engagement/device features
    # plus a country effect, so the demo model reaches a meaningful AUC without
    # the target being trivially separable.
    country_effect = {c: e for c, e in zip(COUNTRIES, rng.normal(0, 0.4, len(COUNTRIES)))}
    countries = rng.choice(COUNTRIES, rows)
    logit = (
        -3.2
        + 0.18 * clicks
        + 0.08 * (memory / 1e9)
        + 0.015 * profile_len
        - 0.06 * impressions
        + 0.05 * sessions
        + np.array([country_effect[c] for c in countries])
        + rng.normal(0, 0.3, rows)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    postback = rng.random(rows) < prob

    df = pd.DataFrame(
        {
            "user_id": [str(uuid.uuid4()) for _ in range(rows)],
            "country": countries,
            "device_os": rng.choice(OSES, rows, p=[0.75, 0.25]),
            "count_user_impressions_7": impressions,
            "appid": rng.choice(app_pool, rows),
            "sdkappid": rng.choice(sdk_pool, rows),
            "memory_total": memory,
            "count_user_clicks_7": clicks,
            "session_count_7d": sessions,
            "user_install_profile": install_profile,
            "postback": postback,
            "timestamp": timestamp,
        }
    )

    # Inject informative missingness, mirroring the real sample.
    df.loc[rng.random(rows) < 0.11, "count_user_clicks_7"] = np.nan
    df.loc[rng.random(rows) < 0.02, "memory_total"] = np.nan
    df.loc[rng.random(rows) < 0.22, "user_install_profile"] = np.nan
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/sample.parquet")
    args = ap.parse_args()

    df = generate(args.rows, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".csv":
        df.to_csv(out, index=False)
    else:
        df.to_parquet(out, index=False)
    print(f"Wrote {len(df)} rows -> {out}  (install rate {df['postback'].mean():.4f})")


if __name__ == "__main__":
    main()
