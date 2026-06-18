import numpy as np


def build_class_weights(labels, n_classes, max_weight=8.0):

    counts = np.bincount(np.asarray(labels).astype(int), minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)  
    weights = np.clip(weights, 1.0, max_weight)
    return weights


class BaselineReward:
    name = "v1_baseline"

    def __init__(self, benign_label=0, n_classes=None, class_weights=None):
        self.benign_label = benign_label

    def __call__(self, action, true_label):
        if action == true_label:
            return 1.0 if true_label == self.benign_label else 2.0
        if true_label == self.benign_label and action != self.benign_label:
            return -1.0   
        if true_label != self.benign_label and action == self.benign_label:
            return -5.0   
        return -0.5       


class ClassWeightedReward:
    name = "v2"

    def __init__(self, benign_label=0, n_classes=None, class_weights=None):
        self.benign_label = benign_label
        self.class_weights = class_weights

    def __call__(self, action, true_label):
        w = float(self.class_weights[true_label])
        if action == true_label:
            if true_label == self.benign_label:
                return 1.0
            return 2.0 * w       
        if true_label == self.benign_label and action != self.benign_label:
            return -1.0          
        if true_label != self.benign_label and action == self.benign_label:
            return -5.0 * w      
        return -0.5 * w          


class AggressiveDetectionReward:
    name = "v3"

    def __init__(self, benign_label=0, n_classes=None, class_weights=None):
        self.benign_label = benign_label
        self.class_weights = class_weights

    def __call__(self, action, true_label):
        w = float(self.class_weights[true_label])
        if action == true_label:
            if true_label == self.benign_label:
                return 1.0
            return 3.0
        if true_label == self.benign_label and action != self.benign_label:
            return -1.0
        if true_label != self.benign_label and action == self.benign_label:
            return -10.0     
        return 0.5          


class PrecisionOrientedReward:
    name = "v4"

    def __init__(self, benign_label=0, n_classes=None, class_weights=None):
        self.benign_label = benign_label
        self.class_weights = class_weights

    def _w(self, true_label):
        return float(self.class_weights[true_label]) if self.class_weights is not None else 1.0

    def __call__(self, action, true_label):
        w = self._w(true_label)
        if action == true_label:
            if true_label == self.benign_label:
                return 2.0      
            return 1.5 * w      
        if true_label == self.benign_label and action != self.benign_label:
            return -8.0      
        if true_label != self.benign_label and action == self.benign_label:
            return -2.0 * w     
        return -1.0 * w        


REWARD_VARIANTS = {
    "v1": BaselineReward,
    "v2": ClassWeightedReward,
    "v3": AggressiveDetectionReward,
    "v4": PrecisionOrientedReward,
}
