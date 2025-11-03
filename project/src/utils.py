# src/utils.py
import os
import numpy as np
import random

def ensure_dirs():
    """Make sure figures and tables directories exist."""
    os.makedirs("figures", exist_ok=True)
    os.makedirs("tables", exist_ok=True)

def set_seed(seed=42):
    """Reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
