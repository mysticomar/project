#!/usr/bin/env python3
"""
Audio Feature Extraction Module
================================
Extracts a 16-dimensional feature vector from audio files using Librosa.

Features extracted:
    - 13 MFCCs (Mel-Frequency Cepstral Coefficients) — timbral texture
    - 1 Tempo (BPM) — rhythmic energy
    - 1 Spectral Centroid — brightness of sound
    - 1 Zero-Crossing Rate — percussiveness / noisiness

Author: Omar Faruk
"""

import librosa
import numpy as np


def extract_features(file_path):
    """
    Extract audio features from a given audio file.

    Loads the audio at 22050 Hz sample rate and computes:
        - 13 MFCCs (mean across time frames)
        - Tempo (beats per minute)
        - Spectral Centroid (mean)
        - Zero-Crossing Rate (mean)

    Args:
        file_path (str): Path to the audio file (.mp3, .wav, .ogg, .m4a).

    Returns:
        np.ndarray: A 16-dimensional feature vector.
    """
    # Load audio file at standard sample rate
    y, sr = librosa.load(file_path, sr=22050)

    # Extract 13 MFCCs and take the mean across time frames
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = np.mean(mfccs.T, axis=0)

    # Extract tempo (beats per minute)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Extract spectral centroid (brightness)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])

    # Extract zero-crossing rate (percussiveness)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])

    # Combine all features into a single vector [13 + 1 + 1 + 1 = 16 dimensions]
    return np.hstack([mfccs, tempo, centroid, zcr])
