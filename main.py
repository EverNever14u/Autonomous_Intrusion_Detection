from tools import *
from reward_callback import MetricLoggerCallback
from plot_generator import (plot_training_metrics, plot_training_comparison,
                            plot_mean_reward)
from data_saver import save_processed_data, load_processed_data, cleaned_data_exists
from reward_functions import REWARD_VARIANTS
from experiments import (train_agent, evaluate_agent,
                         compare_reward_variants, print_comparison_table)

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import joblib
import time
import os


load_dotenv(override=True)

DATA_PATH = os.getenv("DATA_PATH", "data/CICIDS2017_improved/friday.csv")
CLEANED_DIR = os.getenv("CLEANED_DIR", "data/cleaned")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))

TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", 70000))     #base
COMPARE_TIMESTEPS = int(os.getenv("COMPARE_TIMESTEPS", 70000)) #m5

SHOULD_SAVE = str_to_bool(os.getenv("SAVE_DATA", "False"))  
SAVE_MODEL = str_to_bool(os.getenv("SAVE_MODEL", "True"))   
RUN_BASELINE = str_to_bool(os.getenv("RUN_BASELINE", "True"))  
RUN_COMPARE  = str_to_bool(os.getenv("RUN_COMPARE",  "True"))   
LOAD_CLEANED = str_to_bool(os.getenv("LOAD_CLEANED", "True"))



if LOAD_CLEANED and cleaned_data_exists(CLEANED_DIR):
    print(f"{Color.GREEN}Using cleaned data")
    X_train_s, X_test_s, y_train, y_test, le, scaler = load_processed_data(CLEANED_DIR)


    if le is None:
        le_path = os.path.join(MODEL_DIR, "label_encoder.joblib")
        if os.path.exists(le_path):
            le = joblib.load(le_path)
        else:
            print(f"{Color.RED}label_encoder.joblib not found in {CLEANED_DIR} or {MODEL_DIR}.\n"
                  f"Delete {CLEANED_DIR} (or set LOAD_CLEANED=False) to reprocess from the raw file."
                  f"{Color.RESET}")
            exit()

    n_classes = len(le.classes_)
    benign_label = int(np.where(le.classes_ == 'BENIGN')[0][0]) if 'BENIGN' in le.classes_ else 0
    print(f"Classes: {Color.BLUE}{n_classes}{Color.RESET}, "
          f"benign index = {Color.BLUE}{benign_label}{Color.RESET}")

else:
    start_time = time.time()
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found: {DATA_PATH}. Check DATA_PATH.")
        exit()
    data = pd.read_csv(DATA_PATH)
    print(f"Loaded {Color.BLUE}{data.shape[0]}{Color.RESET} rows from "
          f"{os.path.basename(DATA_PATH)} {get_duration(start_time)}")

    start_time = time.time()
    to_drop = ['id', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port',
               'Timestamp', 'Attempted Category']
    data = data.drop(columns=[c for c in to_drop if c in data.columns])
    print(f"Column deletion done {get_duration(start_time)}")

    start_time = time.time()
    data = data.replace([np.inf, -np.inf], np.nan)
    n_before = len(data)
    data = data.dropna().reset_index(drop=True)
    print(f"Dropped {Color.BLUE}{n_before - len(data)}{Color.RESET} rows with inf/nan "
          f"{get_duration(start_time)}")

    start_time = time.time()
    le = LabelEncoder()
    data['Label'] = le.fit_transform(data['Label'])
    n_classes = len(le.classes_)
    benign_label = int(np.where(le.classes_ == 'BENIGN')[0][0]) if 'BENIGN' in le.classes_ else 0
    print(f"Classes encoded: {Color.BLUE}{n_classes}{Color.RESET} classes, "
          f"benign index = {Color.BLUE}{benign_label}{Color.RESET} {get_duration(start_time)}")

    X = data.drop(columns=["Label"])
    y = data["Label"]

    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    print(f"Split + scaling done {get_duration(start_time)}")
    print(f"Training rows: {Color.BLUE}{X_train_s.shape[0]}{Color.RESET}   "
          f"Test rows: {Color.BLUE}{X_test_s.shape[0]}{Color.RESET}")


    if SHOULD_SAVE:
        save_processed_data(X_train_s, X_test_s, y_train, y_test, X.columns, CLEANED_DIR,
                            le=le, scaler=scaler)
    else:
        print(f"Skipped saving")

    if SAVE_MODEL:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
        joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))
        print(f"Saved scaler + label encoder to {Color.BLUE}{MODEL_DIR}/{Color.RESET}")

print(f"{Color.GREEN}\nData processing done{Color.RESET}")
print(f"{Color.GREEN}-{Color.RESET}" * 30 + "\n")



rng = np.random.default_rng(RANDOM_STATE)
val_size = min(5000, len(X_test_s))
val_idx = rng.choice(len(X_test_s), size=val_size, replace=False)
X_val, y_val = X_test_s[val_idx], y_test.values[val_idx]



#base model 
if RUN_BASELINE:
    print(f"{Color.BLUE}base{Color.RESET}")
    reward_fn = REWARD_VARIANTS["v1"](benign_label=benign_label, n_classes=n_classes)
    callback = MetricLoggerCallback(X_val, y_val, reward_fn=reward_fn,
                                    benign_label=benign_label, eval_freq=2000, verbose=1)
    t = time.time()
    baseline_model = train_agent(X_train_s, y_train, n_classes, benign_label, reward_fn,
                                 total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    print(f"Baseline training done {get_duration(t)}")

    plot_mean_reward(callback.timesteps_list, callback.mean_rewards)              
    plot_training_metrics(                                                          
        callback.timesteps_list, callback.accuracies, callback.detection_accuracies,
        title="MILESTONE 4 - accuracy vs attack/benign detection",
        save_path="m4_training_metrics.png"
    )
    evaluate_agent(baseline_model, X_test_s, y_test, le, benign_label,
                   title="base ", print_report=True)

    if SAVE_MODEL:
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = os.path.join(MODEL_DIR, "ppo_baseline")
        baseline_model.save(path)
        print(f"Saved baseline model to {Color.BLUE}{path}.zip{Color.RESET}")


#compare reward variants
best = None
if RUN_COMPARE:
    print(f"\n{Color.BLUE}reward variants{Color.RESET}")
    results, best, best_model = compare_reward_variants(
        X_train_s, y_train, X_test_s, y_test, le, benign_label, n_classes,
        X_val=X_val, y_val=y_val, total_timesteps=COMPARE_TIMESTEPS
    )
    print_comparison_table(results, best)
    plot_training_comparison(results)


    evaluate_agent(best_model, X_test_s, y_test, le, benign_label,
                   title=f"MILESTONE 5 best ({best['name']})", print_report=True)

    if SAVE_MODEL:
        os.makedirs(MODEL_DIR, exist_ok=True)
        for r in results:
            p = os.path.join(MODEL_DIR, f"ppo_{r['name']}")
            r["model"].save(p)
            print(f"Saved agent {r['name']} to {Color.BLUE}{p}.zip{Color.RESET}")
        best_path = os.path.join(MODEL_DIR, f"ppo_best_{best['name']}")
        best_model.save(best_path)
        print(f"{Color.GREEN}Best (by macro-F1): {best['name']} -> {best_path}.zip{Color.RESET}")

print(f"\n{Color.GREEN}All done.{Color.RESET}")
