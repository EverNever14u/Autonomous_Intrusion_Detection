
import os
import sys
import glob
import time
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from stable_baselines3 import PPO

from tools import Color, get_duration
from experiments import evaluate_agent


load_dotenv(override=True)
MODEL_DIR = os.getenv("MODEL_DIR", "models")
CLEANED_DIR = os.getenv("CLEANED_DIR", "data/cleaned")


def find_best_model(model_dir):
    """Prefer an explicit ppo_best_*.zip; otherwise fall back to ppo_baseline.zip."""
    best = sorted(glob.glob(os.path.join(model_dir, "ppo_best_*.zip")))
    if best:
        return best[-1]
    baseline = os.path.join(model_dir, "ppo_baseline.zip")
    return baseline if os.path.exists(baseline) else None


def load_label_encoder():
    """LabelEncoder lives next to the model (models/) or in the cleaned folder."""
    for path in (os.path.join(MODEL_DIR, "label_encoder.joblib"),
                 os.path.join(CLEANED_DIR, "label_encoder.joblib")):
        if os.path.exists(path):
            return joblib.load(path)
    return None


def main():
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        if not os.path.exists(model_path):
            print(f"{Color.RED}Model not found: {model_path}{Color.RESET}")
            return
    else:
        model_path = find_best_model(MODEL_DIR)
        if model_path is None:
            print(f"{Color.RED}No saved model in {MODEL_DIR} "
                  f"(looked for ppo_best_*.zip / ppo_baseline.zip).\n"
                  f"Run `python main.py` with SAVE_MODEL=True first.{Color.RESET}")
            return

    le = load_label_encoder()
    if le is None:
        print(f"{Color.RED}label_encoder.joblib not found in {MODEL_DIR} or {CLEANED_DIR}.\n"
              f"Run `python main.py` (with SAVE_MODEL=True or SAVE_DATA=True) first.{Color.RESET}")
        return
    benign_label = int(np.where(le.classes_ == 'BENIGN')[0][0]) if 'BENIGN' in le.classes_ else 0


    x_test_path = os.path.join(CLEANED_DIR, "X_test.csv")
    y_test_path = os.path.join(CLEANED_DIR, "y_test.csv")
    if not (os.path.exists(x_test_path) and os.path.exists(y_test_path)):
        print(f"{Color.RED}No cleaned test split in {CLEANED_DIR}.\n"
              f"Run `python main.py` once with SAVE_DATA=True to create it.{Color.RESET}")
        return
    X_test = pd.read_csv(x_test_path).values.astype(np.float32)
    y_test = pd.read_csv(y_test_path).squeeze("columns")


    print(f"Model       : {Color.BLUE}{model_path}{Color.RESET}")
    print(f"Test rows   : {Color.BLUE}{X_test.shape[0]}{Color.RESET}")
    print(f"Classes     : {Color.BLUE}{len(le.classes_)}{Color.RESET} "
          f"(benign index = {benign_label})")

    model = PPO.load(model_path, device="cpu")

    t = time.time()
    metrics = evaluate_agent(
        model, X_test, y_test, le, benign_label,
        title=f"TEST - {os.path.basename(model_path)}", print_report=True
    )
    print(f"Evaluation done {get_duration(t)}")

    print(f"\n{Color.GREEN}Summary:{Color.RESET} "
          f"accuracy={metrics['accuracy'] * 100:.2f}%  "
          f"attack_recall={metrics['attack_recall'] * 100:.2f}%  "
          f"macro_F1={metrics['macro_f1'] * 100:.2f}%  "
          f"FP_rate={metrics['fp_rate'] * 100:.2f}%")


if __name__ == "__main__":
    main()
