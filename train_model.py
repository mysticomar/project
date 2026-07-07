#!/usr/bin/env python3
"""
SVM Music Classifier — Training Pipeline
==========================================
Trains a Support Vector Machine (SVM) with a linear kernel to classify
songs into emotion categories based on extracted audio features.

Pipeline:
    1. Load .mp3 files from emotion-labeled directories
    2. Extract 16-dimensional audio features using Librosa
    3. Encode labels and split data (80/20 train/test)
    4. Train SVM classifier with linear kernel
    5. Save trained model, label encoder, and test set as .pkl files

Author: Omar Faruk
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import joblib
from feature_extract import extract_features


# Mapping of emotion labels to their training data directories
class_dirs = {
    'happy': 'happy/',
    'sad': 'sad/',
    'angry': 'angry/',
    'surprise': 'surprise/',
    'disgust': 'disgust/',
    'neutral': 'neutral/'
}


def load_data(class_dirs):
    """
    Load audio files from emotion-labeled directories and extract features.

    Args:
        class_dirs (dict): Mapping of emotion label → directory path.

    Returns:
        tuple: (X, y, le) where:
            - X (np.ndarray): Feature matrix of shape (n_samples, 16)
            - y (np.ndarray): Encoded label array
            - le (LabelEncoder): Fitted label encoder for inverse transforms
    """
    features, labels = [], []
    le = LabelEncoder()
    class_names = list(class_dirs.keys())
    le.fit(class_names)

    for class_name, dir_path in class_dirs.items():
        for file in os.listdir(dir_path):
            if file.endswith('.mp3'):
                feats = extract_features(os.path.join(dir_path, file))
                features.append(feats)
                labels.append(class_name)

    X = np.array(features)
    y = le.transform(labels)
    return X, y, le


# ========================
#  Main Training Pipeline
# ========================
if __name__ == "__main__":
    # Step 1: Load data and extract features
    print("Loading data and extracting features...")
    X, y, le = load_data(class_dirs)
    print(f"Loaded {len(X)} samples across {len(le.classes_)} classes: {list(le.classes_)}")

    # Step 2: Train/test split (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    print(f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    # Step 3: Train SVM with linear kernel
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)

    # Step 4: Evaluate
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:     {test_acc:.4f}")

    # Step 5: Save model artifacts
    joblib.dump((X_test, y_test), "test_set.pkl")
    joblib.dump(clf, "music_classifier_svm.pkl")
    joblib.dump(le, "music_classifier_svm_label_encoder.pkl")
    print("Saved: music_classifier_svm.pkl, music_classifier_svm_label_encoder.pkl, test_set.pkl")
