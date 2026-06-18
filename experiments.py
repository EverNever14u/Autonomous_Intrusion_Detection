import time
import numpy as np
from stable_baselines3 import PPO
from sklearn.metrics import accuracy_score, f1_score

from rl_environment import IDS_Environment
from reward_callback import MetricLoggerCallback
from reward_functions import REWARD_VARIANTS, build_class_weights
from tools import Color, get_duration


def train_agent(X_train, y_train, n_classes, benign_label, reward_fn,
                total_timesteps=50000, learning_rate=3e-4,
                callback=None, shuffle=True, seed=42):
    env = IDS_Environment(X_train, y_train, n_classes=n_classes,
                          benign_label=benign_label, reward_fn=reward_fn, shuffle=shuffle)
    model = PPO("MlpPolicy", env, verbose=0, device="cpu",
                learning_rate=learning_rate, seed=seed)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    return model


def evaluate_agent(model, X_test, y_test, le, benign_label,
                   title="Evaluation", print_report=True):
    y_true = y_test.values if hasattr(y_test, 'values') else np.asarray(y_test).astype(np.int64)
    y_pred, _ = model.predict(X_test, deterministic=True)
    y_pred = np.asarray(y_pred).astype(np.int64)

    n_classes = len(le.classes_)
    labels = list(range(n_classes))

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)

    attack_mask = y_true != benign_label
    benign_mask = ~attack_mask
    attack_recall = float(np.mean(y_pred[attack_mask] != benign_label)) if attack_mask.sum() else float('nan')
    fp_rate = float(np.mean(y_pred[benign_mask] != benign_label)) if benign_mask.sum() else float('nan')

    if print_report:
        print(f"\n{Color.BLUE}=== {title} ==={Color.RESET}")
        print(f"accuracy={accuracy * 100:.2f}%   attack_recall={attack_recall * 100:.2f}%   "
              f"macro_F1={macro_f1 * 100:.2f}%   FP_rate={fp_rate * 100:.2f}%")

    return {"accuracy": accuracy, "attack_recall": attack_recall,
            "macro_f1": macro_f1, "fp_rate": fp_rate}


def compare_reward_variants(X_train, y_train, X_test, y_test, le, benign_label,
                            n_classes, X_val=None, y_val=None,
                            total_timesteps=40000, eval_freq=2000, variant_names=None):
    class_weights = build_class_weights(y_train, n_classes)
    if variant_names is None:
        variant_names = list(REWARD_VARIANTS.keys())

    results = []
    best_metrics = None
    best_model = None

    for name in variant_names:
        reward_fn = REWARD_VARIANTS[name](benign_label=benign_label,
                                          n_classes=n_classes, class_weights=class_weights)

        callback = None
        if X_val is not None and y_val is not None:
            callback = MetricLoggerCallback(X_val, y_val, reward_fn=reward_fn,
                                            benign_label=benign_label,
                                            eval_freq=eval_freq, verbose=0)

        print(f"\n{Color.YELLOW}>>> Training agent: {name}{Color.RESET}")
        t = time.time()
        model = train_agent(X_train, y_train, n_classes, benign_label, reward_fn,
                            total_timesteps=total_timesteps, callback=callback)
        train_time = time.time() - t
        print(f"    trained {get_duration(t)}")

        metrics = evaluate_agent(model, X_test, y_test, le, benign_label,
                                 title=f"Agent {name}", print_report=False)
        metrics["name"] = name
        metrics["train_time_s"] = train_time
        metrics["callback"] = callback
        metrics["model"] = model         
        results.append(metrics)

        if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_metrics = metrics
            best_model = model

    return results, best_metrics, best_model


def print_comparison_table(results, best):
    print(f"\n{Color.BLUE}Agent comparison{Color.RESET}")
    header = f"{'agent':<22}{'acc%':>8}{'atk_rec%':>10}{'macroF1%':>10}{'FP%':>8}{'time_s':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['name']:<22}{r['accuracy'] * 100:>8.2f}{r['attack_recall'] * 100:>10.2f}"
              f"{r['macro_f1'] * 100:>10.2f}{r['fp_rate'] * 100:>8.2f}{r['train_time_s']:>9.1f}")
    print("-" * len(header))
    print(f"{Color.GREEN}Best (by macro-F1): {best['name']}{Color.RESET}")
