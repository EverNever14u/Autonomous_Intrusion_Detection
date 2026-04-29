from tools import *
from plot_generator import generate_plots
from data_saver import save_processed_data

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import shutil
import time
import os


# Data import
start_time = time.time()

load_dotenv(override=True)
PATH = os.getenv("DATA_PATH", "data/CICIDS2017_improved/friday.csv")
CLEANED_DIR = os.getenv("CLEANED_DIR", "data/cleaned")
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))

SHOULD_SAVE = str_to_bool(os.getenv("SAVE_DATA", "True"))
SHOULD_PLOT = str_to_bool(os.getenv("SHOW_PLOTS", "True"))

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
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE, 
    stratify=y
)
print(f"Data division done {get_duration(start_time)}")

print("-" * 30)
print(f"Data processing complete")
print(f"Training rows: {X_train.shape[0]}")
print(f"Test rows: {X_test.shape[0]}")

# saving clean data
if SHOULD_SAVE: save_processed_data(X_train, X_test, y_train, y_test, X.columns, CLEANED_DIR)
else: print(f"{Color.BLUE}Skiped saving (due config){Color.RESET}")
#plots
if SHOULD_PLOT: generate_plots(X, X_scaled, y, le)
else: print(f"{Color.BLUE}Skiped generating plot (due config){Color.RESET}")

print(f"{Color.GREEN}FIN{Color.RESET}")

