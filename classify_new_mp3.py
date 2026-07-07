#!/usr/bin/env python3
"""
Batch Song Classifier
======================
Classifies new .mp3 files using the trained SVM model and automatically
copies them into the appropriate emotion folder in the music library.

Usage:
    1. Drop .mp3 files into the 'songs_to_classify/' directory
    2. Run: python classify_new_mp3.py
    3. Each song is classified and copied to music/<predicted_emotion>/

Author: Omar Faruk
"""

import joblib
import os
import shutil
from feature_extract import extract_features


def load_model():
    """Load the trained SVM classifier and label encoder from disk."""
    classifier = joblib.load("music_classifier_svm.pkl")
    label_encoder = joblib.load("music_classifier_svm_label_encoder.pkl")
    return classifier, label_encoder


def classify_mp3(file_path, classifier, label_encoder):
    """
    Classify a single audio file into an emotion category.

    Args:
        file_path (str): Path to the .mp3 file.
        classifier: Trained SVM model.
        label_encoder: Fitted LabelEncoder.

    Returns:
        str: Predicted emotion label (e.g., 'happy', 'sad', etc.)
    """
    feats = extract_features(file_path)
    pred = classifier.predict([feats])[0]
    return label_encoder.inverse_transform([pred])[0]


if __name__ == "__main__":
    classifier, label_encoder = load_model()
    input_dir = 'songs_to_classify/'
    output_base = 'music'

    if not os.path.isdir(input_dir):
        print(f"Error: '{input_dir}' directory not found. Run mkdirs.py first.")
        exit(1)

    mp3_files = [f for f in os.listdir(input_dir) if f.endswith('.mp3')]

    if not mp3_files:
        print(f"No .mp3 files found in '{input_dir}'.")
        exit(0)

    print(f"Classifying {len(mp3_files)} song(s)...\n")

    for file in mp3_files:
        file_path = os.path.join(input_dir, file)
        result = classify_mp3(file_path, classifier, label_encoder)
        dest = os.path.join(output_base, result)
        shutil.copy(file_path, dest)
        print(f"  {file}  →  {result}/")

    print(f"\nDone! All songs classified and copied to '{output_base}/'.")
