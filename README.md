# YoungEverest — Bike Sharing Demand ] (Reproducible Code)


This repository contains all preprocessing, feature engineering, classical ML baselines, and MLflow tracking required to fully reproduce the midpoint results for the Bike Sharing Demand project (CS-4120 Machine Learning).

All training logic is located inside the `src/` directory.   
No raw data files or MLflow artifacts are committed.

---

## Environment

## ⚙️ Installation

## Python Version

This project is tested and reproducible on:

**Python 3.10 or Python 3.11**

Newer versions (e.g., Python 3.12 or 3.13) may cause dependency failures because 
NumPy, scikit-learn, and MLflow do not yet fully support them.

Please ensure you run the project in a Python 3.10/3.11 environment.
I had python 3.13 on my device so i had to create a tf_env and activate the environment so i couls use python 3.10 and run the models.

 - clone the repo to your terminal
 - go to Bike-Sharing-Demand folder
```bash
pip install -r requirements.txt

to run baseline:
python src/train_baselines.py
mlflow ui
then open:
http://127.0.0.1:5000

To run the final neural network models:

python3 src/train_nn.py

This will:
- preprocess the data using the same ColumnTransformer
- train the tuned NN classifier and regressor
- save models in models/
- log metrics to MLflow

AI Assistance Disclosure

AI tools (ChatGPT) were used only for clarifying concepts and suggesting code structure.
All preprocessing, feature engineering, model training, evaluation, and plotting were implemented manually by the authors.
Final code and analysis reflect our own work and understanding.

