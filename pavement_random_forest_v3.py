"""
Random Forest regression using one or more input CSV files provided at runtime.

Usage (single file):
    python pavement_random_forest_v3.py Data_Texas_No_Missing.csv

Usage (multiple files - train/evaluate each separately):
    python pavement_random_forest_v3.py Data_Texas_No_Missing.csv Data_Alabama_No_Missing.csv

Usage (multiple files - merge all into one dataset):
    python pavement_random_forest_v3.py Data_Texas_No_Missing.csv Data_Alabama_No_Missing.csv --merge

CSV column order (1-indexed):
1 YEAR
2 SN
3 PRECIPITATION
4 TEMPERATURE
5 IRI                 <- label (regression target)
6 INITIAL_IRI
7 STATION_ID
8 AGE
9 ACCUMULATED_AADTT

Features (2,3,4,6,8,9): SN, PRECIPITATION, TEMPERATURE, INITIAL_IRI, AGE, ACCUMULATED_AADTT
Label (5): IRI
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
FEATURE_COLS = ["SN", "PRECIPITATION", "TEMPERATURE", "INITIAL_IRI", "AGE", "ACCUMULATED_AADTT"]
LABEL_COL    = "IRI"
EXPECTED_COLS = [
    "YEAR", "SN", "PRECIPITATION", "TEMPERATURE", "IRI",
    "INITIAL_IRI", "STATION_ID", "AGE", "ACCUMULATED_AADTT"
]


# ─────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Random Forest model on one or more pavement CSV files."
    )
    parser.add_argument(
        "csv_files",
        nargs="+",                          # Accept one or more files
        help="Input CSV file(s), e.g., Data_Texas.csv Data_Alabama.csv",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all CSV files into a single dataset before training (default: train each file separately).",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# CSV loading & preprocessing
# ─────────────────────────────────────────────
def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path.name} is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    work = df[FEATURE_COLS + [LABEL_COL]].copy()
    for c in FEATURE_COLS + [LABEL_COL]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna().reset_index(drop=True)
    return work


# ─────────────────────────────────────────────
# Model training & evaluation
# ─────────────────────────────────────────────
def train_and_evaluate(work: pd.DataFrame, label: str) -> None:
    X = work[FEATURE_COLS].to_numpy(dtype=float)
    y = work[LABEL_COL].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=4,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)
    print(f"  Samples used : {len(work)}")
    print(f"  Train size   : {len(y_train)}")
    print(f"  Test size    : {len(y_test)}")
    print(f"  R^2          : {r2:.4f}")
    print(f"  RMSE         : {rmse:.4f}")
    print(f"\n  Feature importance:")
    for name, val in zip(FEATURE_COLS, model.feature_importances_):
        print(f"    {name:20s}: {val:.6f}")
    print()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    csv_paths = [Path(f) for f in args.csv_files]

    if args.merge:
        # ── Merge all files into a single dataset and train ──
        frames = []
        for p in csv_paths:
            df = load_csv(p)
            print(f"Loaded {p.name}: {len(df)} rows")
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        label = f"MERGED ({', '.join(p.name for p in csv_paths)})"
        train_and_evaluate(combined, label)

    else:
        # ── Train and evaluate each file independently ──
        for p in csv_paths:
            df = load_csv(p)
            train_and_evaluate(df, p.name)


if __name__ == "__main__":
    main()
