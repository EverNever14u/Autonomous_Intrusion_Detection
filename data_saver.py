import os
import shutil
import time
import joblib
import numpy as np
import pandas as pd
from tools import get_duration, Color


_REQUIRED_FILES = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]


def save_processed_data(X_train, X_test, y_train, y_test, columns, directory,
                        le=None, scaler=None):
    start_time = time.time()

    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    print(f"Folder {directory} cleared and recreated {get_duration(start_time)}")

    X_train_df = pd.DataFrame(X_train, columns=columns)
    X_test_df = pd.DataFrame(X_test, columns=columns)

    save_start = time.time()
    X_train_df.to_csv(f"{directory}/X_train.csv", index=False)
    X_test_df.to_csv(f"{directory}/X_test.csv", index=False)
    y_train.to_csv(f"{directory}/y_train.csv", index=False)
    y_test.to_csv(f"{directory}/y_test.csv", index=False)

    if le is not None:
        joblib.dump(le, os.path.join(directory, "label_encoder.joblib"))
    if scaler is not None:
        joblib.dump(scaler, os.path.join(directory, "scaler.joblib"))

    print(f"All files saved in {directory} {get_duration(save_start)}")
    print("-" * 30)


def cleaned_data_exists(directory):
    """True only if the four CSVs of a complete split are present."""
    return all(os.path.exists(os.path.join(directory, f)) for f in _REQUIRED_FILES)


def load_processed_data(directory):
    start_time = time.time()
    X_train = pd.read_csv(os.path.join(directory, "X_train.csv")).values.astype(np.float32)
    X_test = pd.read_csv(os.path.join(directory, "X_test.csv")).values.astype(np.float32)
    y_train = pd.read_csv(os.path.join(directory, "y_train.csv")).squeeze("columns")
    y_test = pd.read_csv(os.path.join(directory, "y_test.csv")).squeeze("columns")

    le_path = os.path.join(directory, "label_encoder.joblib")
    scaler_path = os.path.join(directory, "scaler.joblib")
    le = joblib.load(le_path) if os.path.exists(le_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    print(f"Loaded cleaned data from {Color.BLUE}{directory}{Color.RESET} {get_duration(start_time)}")
    print(f"  Training rows: {Color.BLUE}{X_train.shape[0]}{Color.RESET}   "
          f"Test rows: {Color.BLUE}{X_test.shape[0]}{Color.RESET}")
    return X_train, X_test, y_train, y_test, le, scaler
