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


def extract_hog(image):
    hog_features, _ = feature.hog(image, visualize=True)
    return hog_features

def load_images_from_folder(folder):
    images = []
    labels = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, filename)
                if img_path.endswith(".png") or img_path.endswith(".bmp"):
                    img = io.imread(img_path)
                    img_gray = color.rgb2gray(img) if img.ndim == 3 else img
                    hog_features = extract_hog(img_gray)
                    images.append(hog_features)
                    labels.append(subfolder)
    return np.array(images), np.array(labels)

data_folder = "complete_ms_data"
X, y = load_images_from_folder(data_folder)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
