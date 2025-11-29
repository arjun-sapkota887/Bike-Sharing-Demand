# YoungEverest — Bike Sharing Demand ] (Reproducible Code)


This repository contains all preprocessing, feature engineering, classical ML baselines, and MLflow tracking required to fully reproduce the midpoint results for the Bike Sharing Demand project (CS-4120 Machine Learning).

All training logic is located inside the `src/` directory.  
All EDA is inside `notebooks/`.  
No raw data files or MLflow artifacts are committed.

---


project/
│
├── data/
│   ├── README.md               # download instructions (no raw CSVs stored)
│
│
├── mlruns/                     # MLflow tracking directory (ignored in git)
│
├── notebooks/
│   └── midpoint_notebook.ipynb # EDA only; not used for training
│
├── src/
│   ├── data.py                 # loading, cleaning, splitting
│   ├── features.py             # feature engineering and label creation
│   ├── utils.py                # MLflow logger + helpers
│   ├── train_baselines.py      # classical ML baselines
│   ├── train_nn.py             # placeholder for final NN
│   └── evaluate.py             # plots & evaluation
│
├── README.md                   # THIS file
├── requirements.txt
└── .gitignore

## 📁 Project Structure

## Environment

## ⚙️ Installation

From inside the `project/` folder:

```bash
pip install -r requirements.txt

to run baseline:
python src/train_baselines.py
mlflow ui
