from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

class RewardLoggerCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=1000, n_eval_episodes=5, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.mean_rewards = []
        self.std_rewards = []
        self.timesteps_list = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True
            )
            self.mean_rewards.append(mean_reward)
            self.std_rewards.append(std_reward)
            self.timesteps_list.append(self.num_timesteps)

            if self.verbose:
                print(f"  [Eval @ {self.num_timesteps} steps] "
                      f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
        return True