# src/features.py
import numpy as np
import pandas as pd

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Feels-like gap captures calibration differences
    df['feels_like_gap'] = (df['atemp'] - df['temp']).astype(float)

    # 2) Rush hour flag (commute peaks beyond naive threshold)
    df['rush_hour'] = df['hour'].isin([7,8,9,16,17,18]).astype(int)

    # (Optional bonus) seasonality encoding via sine/cosine of hour
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
    return df

def feature_list():
    return ['temp','atemp','humidity','windspeed',
            'hour','weekday','month','year',
            'feels_like_gap','rush_hour','hour_sin','hour_cos']
