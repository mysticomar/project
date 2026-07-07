#!/usr/bin/env python3
"""
Confusion Matrix Generator
============================
Loads the trained SVM model and test set, generates predictions, and
creates a publication-ready confusion matrix heatmap.

Output: confusion_matrix.png (300 DPI)

Author: Omar Faruk
"""

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


def main():
    """Generate confusion matrix from saved model and test data."""
    # Load saved artifacts
    X_test, y_test = joblib.load("test_set.pkl")
    le = joblib.load("music_classifier_svm_label_encoder.pkl")
    clf = joblib.load("music_classifier_svm.pkl")

    # Generate predictions
    y_pred = clf.predict(X_test)

    # Print evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        cmap='Blues',
        linewidths=0.5,
        square=True
    )
    plt.title('Music Emotion Classification — Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()

    # Save the figure
    output_path = 'confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved confusion matrix heatmap → {output_path}")


if __name__ == "__main__":
    main()
