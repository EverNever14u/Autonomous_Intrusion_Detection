from gymnasium import spaces
import gymnasium as gym
import numpy as np

from reward_functions import BaselineReward


class IDS_Environment(gym.Env):
    def __init__(self, features, labels, n_classes=None, benign_label=0,
                 reward_fn=None, shuffle=False):
        super(IDS_Environment, self).__init__()

        feats = features.values if hasattr(features, 'values') else features
        labs = labels.values if hasattr(labels, 'values') else labels

        self.features = np.asarray(feats, dtype=np.float32)
        self.labels = np.asarray(labs).astype(np.int64)

        self.n_samples = len(self.features)
        self.n_features = self.features.shape[1]


        self.n_classes = n_classes if n_classes is not None else len(np.unique(self.labels))


        self.benign_label = benign_label


        self.reward_fn = reward_fn if reward_fn is not None else BaselineReward(benign_label=benign_label)

        self.shuffle = shuffle
        self.current_step = 0
        self.order = np.arange(self.n_samples)

        self.action_space = spaces.Discrete(self.n_classes)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_features,), dtype=np.float32
        )

    def step(self, action):
        idx = self.order[self.current_step]
        true_label = self.labels[idx]

        reward = float(self.reward_fn(int(action), int(true_label)))

        self.current_step += 1
        terminated = self.current_step >= self.n_samples
        truncated = False

        if terminated:
            obs = np.zeros(self.n_features, dtype=np.float32)
        else:
            obs = self.features[self.order[self.current_step]]

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        if self.shuffle:
            self.order = self.np_random.permutation(self.n_samples)
        else:
            self.order = np.arange(self.n_samples)
        obs = self.features[self.order[0]]
        return obs, {}
