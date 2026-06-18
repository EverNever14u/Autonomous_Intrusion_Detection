import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MetricLoggerCallback(BaseCallback):
    """
    Every `eval_freq` steps, evaluate the policy on a small FIXED validation
    subset with ONE batched prediction, and log:

      - mean_reward        : average reward PER SAMPLE (same reward_fn as
                             training). Fixed denominator -> comparable across
                             evaluations -> this is the M4 mean_reward plot.
      - accuracy           : EXACT multi-class accuracy (right class chosen).
      - detection_accuracy : BINARY "attack vs benign" accuracy. Did the model
                             get attack-or-not right, IGNORING the exact attack
                             type? (a wrong attack type still counts as a correct
                             detection here). This is always >= exact accuracy.
    """

    def __init__(self, X_val, y_val, reward_fn, benign_label=0, eval_freq=2000, verbose=1):
        super().__init__(verbose)
        self.X_val = np.asarray(X_val, dtype=np.float32)
        self.y_val = np.asarray(y_val).astype(np.int64)
        self.reward_fn = reward_fn
        self.benign_label = benign_label
        self.eval_freq = eval_freq

        self.timesteps_list = []
        self.mean_rewards = []
        self.accuracies = []
        self.detection_accuracies = []

        # ground truth collapsed to binary attack(True)/benign(False), once
        self._y_true_is_attack = self.y_val != self.benign_label

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            preds, _ = self.model.predict(self.X_val, deterministic=True)
            preds = np.asarray(preds).astype(np.int64)

            # exact multi-class accuracy
            accuracy = float(np.mean(preds == self.y_val))

            # binary attack/not-attack accuracy (ignores the exact category)
            pred_is_attack = preds != self.benign_label
            detection_accuracy = float(np.mean(pred_is_attack == self._y_true_is_attack))

            # per-sample mean reward, using the exact same reward as training
            rewards = [self.reward_fn(int(p), int(t)) for p, t in zip(preds, self.y_val)]
            mean_reward = float(np.mean(rewards))

            self.timesteps_list.append(self.num_timesteps)
            self.mean_rewards.append(mean_reward)
            self.accuracies.append(accuracy)
            self.detection_accuracies.append(detection_accuracy)

            if self.verbose:
                print(f"  [Eval @ {self.num_timesteps} steps] "
                      f"mean_reward={mean_reward:.3f}  "
                      f"accuracy={accuracy * 100:.2f}%  "
                      f"detection(attack/benign)={detection_accuracy * 100:.2f}%")
        return True
