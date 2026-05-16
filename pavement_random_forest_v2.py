"""
Random Forest regression using an input CSV file provided at runtime.

Usage:
    python pavement_random_forest.py Data_Texas_No_Missing.csv

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Random Forest regression model using a pavement CSV file."
    )
    parser.add_argument(
        "csv_file",
        help="Input CSV file, e.g., Data_Texas_No_Missing.csv",
    )
    return parser.parse_args()


args = parse_args()
csv_path = Path(args.csv_file)

if not csv_path.exists():
    raise FileNotFoundError(f"Input CSV file not found: {csv_path}")

# -----------------------------
# 1) Load CSV
# -----------------------------
df = pd.read_csv(csv_path)

# Make column names robust (strip spaces)
df.columns = [c.strip() for c in df.columns]

# Expected headers (as user provided)
expected = [
    "YEAR", "SN", "PRECIPITATION", "TEMPERATURE", "IRI",
    "INITIAL_IRI", "STATION_ID", "AGE", "ACCUMULATED_AADTT"
]
missing = [c for c in expected if c not in df.columns]
if missing:
    raise ValueError(
        f"{csv_path.name} is missing required columns: {missing}\n"
        f"Found columns: {list(df.columns)}"
    )

# -----------------------------
# 2) Select features/label
# -----------------------------
feature_cols = ["SN", "PRECIPITATION", "TEMPERATURE", "INITIAL_IRI", "AGE", "ACCUMULATED_AADTT"]
label_col = "IRI"

# Keep only needed columns, drop rows with missing values
work = df[feature_cols + [label_col]].copy()
work = work.dropna().reset_index(drop=True)

# Ensure numeric
for c in feature_cols + [label_col]:
    work[c] = pd.to_numeric(work[c], errors="coerce")
work = work.dropna().reset_index(drop=True)

X = work[feature_cols].to_numpy(dtype=float)
y = work[label_col].to_numpy(dtype=float)

# -----------------------------
# 3) Train/test split (75/25)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -----------------------------
# 4) Train Random Forest regressor
# -----------------------------
# Match the same general experimental setup: random_state=42 and parallel jobs.
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=4,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# -----------------------------
# 5) Evaluate
# -----------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Input file  : {csv_path.name}")
print(f"Samples used: {len(work)}")
print(f"Train size  : {len(y_train)}")
print(f"Test size   : {len(y_test)}")
print(f"R^2         : {r2:.4f}")
print(f"RMSE        : {rmse:.4f}")

# -----------------------------
# 6) Feature importance (optional)
# -----------------------------
# Random Forest uses impurity-based feature importance.
importances = model.feature_importances_

print("\nFeature importance:")
for name, val in zip(feature_cols, importances):
    print(f"  {name:18s}: {val:.6f}")
