# Pavement IRI Prediction — Usage Guide

Three regression models are available. All scripts share the same interface.

| Script | Model |
|---|---|
| `pavement_random_forest_v3.py` | Random Forest |
| `pavement_xgboost_v3.py` | XGBoost |
| `pavement_light_gbm_v3.py` | LightGBM |

---

## Requirements

Install the required Python packages before running:

```bash
pip install numpy pandas scikit-learn xgboost lightgbm
```

---

## Input CSV Format

Each CSV file must contain the following columns (column names are case-sensitive):

| # | Column Name | Role |
|---|---|---|
| 1 | `YEAR` | — |
| 2 | `SN` | Feature |
| 3 | `PRECIPITATION` | Feature |
| 4 | `TEMPERATURE` | Feature |
| 5 | `IRI` | **Label (target)** |
| 6 | `INITIAL_IRI` | Feature |
| 7 | `STATION_ID` | — |
| 8 | `AGE` | Feature |
| 9 | `ACCUMULATED_AADTT` | Feature |

---

## Usage

### 1. Single file

Train and evaluate on one CSV file.

```bash
python pavement_random_forest_v3.py Data_Texas_No_Missing.csv
python pavement_xgboost_v3.py Data_Texas_No_Missing.csv
python pavement_light_gbm_v3.py Data_Texas_No_Missing.csv
```

---

### 2. Multiple files — evaluated separately (default)

Pass two or more CSV files. Each file is trained and evaluated independently,
and results are printed one after another.

```bash
python pavement_random_forest_v3.py Data_Alabama_No_Missing.csv Data_Texas_No_Missing.csv
python pavement_xgboost_v3.py Data_Alabama_No_Missing.csv Data_Texas_No_Missing.csv
python pavement_light_gbm_v3.py Data_Alabama_No_Missing.csv Data_Texas_No_Missing.csv
```

---

### 3. Multiple files — merged into one dataset (`--merge`)

All CSV files are concatenated into a single dataset before training.
One model is trained and one result is printed.

```bash
python pavement_random_forest_v3.py Data_Alabama_No_Missing.csv Data_Texas_No_Missing.csv --merge
python pavement_xgboost_v3.py Data_Alabama_No_Missing.csv Data_Texas_No_Missing.csv --merge
python pavement_light_gbm_v3.py Data_Alabama_No_Missing.csv Data_Texas_No_Missing.csv --merge
```

---

## Output

Each run prints the following metrics for every dataset evaluated:

```
=======================================================
  Data_Texas_No_Missing.csv
=======================================================
  Samples used : 1200
  Train size   : 900
  Test size    : 300
  R^2          : 0.9123
  RMSE         : 0.1847

  Feature importance (gain):
    SN                  : 0.123456
    PRECIPITATION       : 0.045678
    TEMPERATURE         : 0.031234
    INITIAL_IRI         : 0.512345
    AGE                 : 0.234567
    ACCUMULATED_AADTT   : 0.052720
```

---

## Mode Comparison

| Mode | Command | Behavior |
|---|---|---|
| Single file | `script.py A.csv` | Train on A, evaluate on A |
| Separate | `script.py A.csv B.csv` | Train on A → evaluate, then train on B → evaluate |
| Merged | `script.py A.csv B.csv --merge` | Combine A+B, train once, evaluate once |
