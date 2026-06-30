"""Reproducible data analysis used to design the pipeline.

Prints the profile that drove the feature-engineering and modelling decisions:
target balance, missingness, cardinality, distributions, the temporal window and
simple signal checks. Works on parquet or csv.

Usage:  python scripts/eda.py --data data/sample.parquet
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def run(path: str) -> None:
    df = pd.read_parquet(path) if path.endswith((".parquet", ".pq")) else pd.read_csv(path)
    y = df["postback"]
    y = (y.astype(str).str.lower().map({"true": 1, "false": 0}).fillna(y)
         if y.dtype == object else y.astype(int))
    y = y.astype(int)

    print(f"rows={len(df):,}  cols={df.shape[1]}")
    print(f"install base-rate = {y.mean():.4f}  (positives={int(y.sum())})\n")

    print("== missingness ==")
    print((df.isnull().mean().sort_values(ascending=False) * 100).round(2).astype(str) + " %")

    print("\n== cardinality (string fields) ==")
    for c in ["user_id", "country", "device_os", "appid", "sdkappid"]:
        if c in df:
            print(f"  {c:24s} nunique={df[c].nunique():>6}  ({df[c].nunique()/len(df):.1%} of rows)")

    print("\n== numeric summary ==")
    for c in ["count_user_impressions_7", "count_user_clicks_7", "session_count_7d", "memory_total"]:
        if c in df:
            s = pd.to_numeric(df[c], errors="coerce")
            print(f"  {c:26s} p50={s.median():>12.1f} p99={s.quantile(.99):>12.1f} max={s.max():>14.1f} null={s.isna().mean():.1%}")

    print("\n== temporal window ==")
    ts = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms", utc=True)
    print(f"  {ts.min()} -> {ts.max()}  span={ (ts.max()-ts.min()).days } days")

    print("\n== informative missingness (install rate by null-ness) ==")
    for c in ["count_user_clicks_7", "user_install_profile"]:
        if c in df:
            m = df[c].isna()
            print(f"  {c}: null->{y[m].mean():.4f}  present->{y[~m].mean():.4f}")

    if "user_install_profile" in df:
        L = df["user_install_profile"].apply(lambda x: 0 if pd.isna(x) else len(str(x).split()))
        print(f"\n== install profile length: p50={L.median():.0f} p99={L.quantile(.99):.0f} max={L.max()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    run(ap.parse_args().data)
