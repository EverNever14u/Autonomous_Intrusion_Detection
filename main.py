import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ANSI color codes
class Color:
    GREEN = '\033[92m'
    RESET = '\033[0m'

# Function to format time in green
def get_duration(start):
    duration_s = (time.time() - start)
    return f"{Color.GREEN}({duration_s:.2f}s){Color.RESET}"

# Data import
start_time = time.time()
PATH = "data/CICIDS2017_improved/friday.csv" 
try:
    data = pd.read_csv(PATH)
    print(f"Loaded {data.shape[0]} rows {get_duration(start_time)}.")
except FileNotFoundError:
    print("File not found. Please check your PATH.")
    exit()

# Delete meaningless columns
start_time = time.time()
to_drop = ['id', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp', 'Attempted Category']
data = data.drop(columns=[column for column in to_drop if column in data.columns])
print(f"Column deletion done {get_duration(start_time)}")

# Label encoding
start_time = time.time()
le = LabelEncoder()
data['Label'] = le.fit_transform(data['Label'])
print(f"Classes encoded {get_duration(start_time)}")

# Feature/Target separation
start_time = time.time()
X = data.drop('Label', axis=1)
y = data['Label']
print(f"Dropping done {get_duration(start_time)}")

# Scaling
start_time = time.time()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Scaling done {get_duration(start_time)}")

# Data division
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)
print(f"Data division done {get_duration(start_time)}")

print("-" * 30)
print(f"Data processing complete")
print(f"Training rows: {X_train.shape[0]}")
print(f"Test rows: {X_test.shape[0]}")