#	loading CSV
# ---- Load data ----
df = pd.read_csv("train.csv")
print("Raw shape:", df.shape)
df.head()

#cleaning and datetime parsing

# Convert datetime
df['datetime'] = pd.to_datetime(df['datetime'])

# Drop leakage columns: casual + registered sum to count
df = df.drop(columns=['casual', 'registered'])

# Extract time features
df['hour']    = df['datetime'].dt.hour
df['weekday'] = df['datetime'].dt.weekday
df['month']   = df['datetime'].dt.month
df['year']    = df['datetime'].dt.year

print("After cleaning:", df.shape)
df.head()

#splitting function (train/val/test)
eature_cols = [
    'season', 'holiday', 'workingday', 'weather',
    'temp', 'atemp', 'humidity', 'windspeed',
    'hour', 'weekday', 'month', 'year',
    'feels_like_gap', 'rush_hour', 'hour_sin', 'hour_cos'
]

X = df[feature_cols]
y_class = df['is_peak_hour']
y_reg = df['count']

# First split: train (70%), temp (30%)
X_train, X_temp, y_class_train, y_class_temp, y_reg_train, y_reg_temp = train_test_split(
    X, y_class, y_reg,
    test_size=0.30,
    random_state=42,
    stratify=y_class
)

# Second split: val (15%), test (15%)
X_val, X_test, y_class_val, y_class_test, y_reg_val, y_reg_test = train_test_split(
    X_temp, y_class_temp, y_reg_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_class_temp
)

print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)

