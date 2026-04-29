from gymnasium import spaces
import gymnasium as gym
import numpy as np

class IDS_Environment(gym.Env):
    def __init__(self, features, labels):
        super(IDS_Environment, self).__init__()
        
        self.features = features.values if hasattr(features, 'values') else features
        self.labels = labels.values if hasattr(labels, 'values') else labels
        
        self.n_samples = len(self.features)
        self.n_features = self.features.shape[1]
        self.n_classes = len(np.unique(self.labels))
        
        self.current_step = 0

        #action space:
        self.action_space = spaces.Discrete(self.n_classes)
        
        #observation space:
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.n_features,), 
            dtype=np.float32
        )

    def step(self, action):
        true_label = self.labels[self.current_step]
        
        #reward function
        reward = self._calculate_reward(action, true_label)
        
        self.current_step += 1
        
        terminated = self.current_step >= self.n_samples
        truncated = False
        
        obs = self.features[self.current_step] if not terminated else np.zeros(self.n_features)
        
        return obs, reward, terminated, truncated, {}

    def _calculate_reward(self, action, true_label):
        # good ans
        if action == true_label:
            if true_label == 0:
                return 1.0  # pass normal traffic
            else:
                return 2.0  # block attack
                
        # bad ans
        else:
            if true_label == 0 and action != 0:
                return -1.0 # False Positive 
            elif true_label != 0 and action == 0:
                return -5.0 # False Negative 
            else:
                return -0.5 # wrong type of attack

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self.features[self.current_step], {}