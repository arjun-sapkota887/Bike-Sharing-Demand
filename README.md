# YoungEverest — Bike Sharing Demand (Midpoint)

Reproduces baselines for **classification** (is_peak_hour) and **regression** (count) on the Kaggle Bike Sharing dataset.  
Midpoint artifacts (4 plots + 2 tables) are written to `project/figures` and `project/tables`.

## Dataset
Kaggle: Bike Sharing Demand. Place `train.csv` under `project/data/`.  
We remove `casual` and `registered` to avoid leakage (they sum to `count`).

## Environment
```bash
to install dependencies :
pip install -r requirements.txt

to run baseline:
python src/train_baselines.py
mlflow ui
