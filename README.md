# IRI Prediction with XGBoost

This repo contains a dataset (`data.csv`) and Python code to predict pavement roughness (IRI) using an XGBoost regression model.

## Data
- Samples: 612  
- Features (6): `SN`, `PRECIPITATION`, `TEMPERATURE`, `INITIAL_IRI`, `AGE`, `ACCUMULATED_AADTT`  
- Label: `IRI`

## Run
```bash
pip install xgboost scikit-learn pandas numpy
python pavement_xgboost.py
