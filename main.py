import os
import numpy as np
from skimage import io, color, feature
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder

class NaiveBayesClassifier:
    def __init__(self):
        self.class_stats = {}

    def fit(self, X, y):
        unique_classes = np.unique(y)
        for cls in unique_classes:
            cls_data = X[y == cls]
            cls_mean = np.mean(cls_data, axis=0)
            cls_std = np.std(cls_data, axis=0)
            self.class_stats[cls] = {'mean': cls_mean, 'std': cls_std}

    def predict(self, X):
        num_classes = len(self.class_stats)
        probabilities = np.zeros((X.shape[0], num_classes))

        for i, cls in enumerate(self.class_stats):
            prior_prob = len(X_train[y_train == cls]) / len(X_train)
            likelihood = norm.pdf(X, loc=self.class_stats[cls]['mean'], scale=self.class_stats[cls]['std'])
            probabilities[:, i] = prior_prob * np.prod(likelihood, axis=1)

        return probabilities