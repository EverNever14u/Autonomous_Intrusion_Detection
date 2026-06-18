import os
import matplotlib.pyplot as plt
import numpy as np


PLOTS_DIR = os.getenv("PLOTS_DIR", "plots")


def _plot_path(save_path):
    head = os.path.dirname(save_path)
    if head:                      
        os.makedirs(head, exist_ok=True)
        return save_path
    os.makedirs(PLOTS_DIR, exist_ok=True)
    return os.path.join(PLOTS_DIR, save_path)


def plot_mean_reward(timesteps, mean_rewards, save_path='mean_reward_plot.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, mean_rewards, color='green', marker='o', label='Mean reward / sample')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('Timesteps')
    plt.ylabel('Mean reward per sample')
    plt.title('Mean reward during training')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(_plot_path(save_path), dpi=150)
    plt.show()
    plt.close()


def plot_training_metrics(timesteps, accuracies, detection_accuracies,
                          title='Validation metrics during training',
                          save_path='training_metrics_plot.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, [a * 100 for a in accuracies],
             color='blue', marker='o', label='Accuracy (exact class)')
    plt.plot(timesteps, [a * 100 for a in detection_accuracies],
             color='red', marker='s', label='Detection (attack vs benign)')
    plt.xlabel('Timesteps')
    plt.ylabel('%')
    plt.title(title)
    plt.ylim(0, 101)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(_plot_path(save_path), dpi=150)
    plt.show()
    plt.close()


def plot_training_comparison(results, save_path='m5_comparison_curves.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(results), 1)))

    for r, c in zip(results, colors):
        cb = r.get('callback')
        if cb is None:
            continue
        ts = cb.timesteps_list
        ax1.plot(ts, [a * 100 for a in cb.accuracies], marker='o', color=c, label=r['name'])
        ax2.plot(ts, [a * 100 for a in cb.detection_accuracies], marker='s', color=c, label=r['name'])

    ax1.set_title('Accuracy (exact class)')
    ax2.set_title('Detection (attack vs benign)')
    for ax in (ax1, ax2):
        ax.set_xlabel('Timesteps')
        ax.set_ylim(0, 101)
        ax.grid(True)
        ax.legend()
    ax1.set_ylabel('%')
    fig.suptitle('M5 - reward variants comparison')
    fig.tight_layout()
    fig.savefig(_plot_path(save_path), dpi=150)
    plt.show()
    plt.close(fig)
