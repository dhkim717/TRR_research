# IRI Prediction with XGBoost and LightGBM

This repository provides a dataset (`data.csv`) and baseline machine learning scripts to predict pavement roughness (International Roughness Index, IRI) using gradient boosting regression models.

## Dataset

- File: `data.csv`
- Samples: 612
- Target (label): `IRI`
- Features (6):
  - `SN`
  - `PRECIPITATION`
  - `TEMPERATURE`
  - `INITIAL_IRI`
  - `AGE`
  - `ACCUMULATED_AADTT`

## Repository Structure

- `data.csv` : dataset
- `pavement_xgboost.py` : XGBoost regression baseline
- `pavement_light_gbm.py` : LightGBM regression baseline

## Requirements

- Python 3.8+ recommended

## Run: XGBoost

Install dependencies:

```bash
pip install xgboost scikit-learn pandas numpy
python pavement_xgboost.py

## Run: Light GBM
pip install lightgbm
python pavement_light_gbm.py
