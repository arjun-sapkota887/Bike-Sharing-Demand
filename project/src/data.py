# src/data.py
import pandas as pd

def load_data(csv_path: str):
    """Load and minimally clean the Bike Sharing dataset."""
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Drop leakage columns (casual + registered)
    for col in ['casual', 'registered']:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Extract useful datetime parts
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    
    return df
