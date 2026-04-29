import os
import shutil
import pandas as pd
import time
from tools import get_duration

def save_processed_data(X_train, X_test, y_train, y_test, columns, directory):
    """
    save processed data
    """
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

    print(f"All files saved in {directory} {get_duration(save_start)}")
    print("-" * 30)