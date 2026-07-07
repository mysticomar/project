#!/usr/bin/env python3
"""
Directory Setup Script
=======================
Creates the required folder structure for the Emotion-Based Music
Recommendation System.

Directories created:
    Training data:    happy/, sad/, angry/, surprise/, disgust/, neutral/
    Music playback:   music/happy/, music/sad/, ... music/neutral/
    Classification:   songs_to_classify/

Author: Omar Faruk
"""

import os


def setup_directories():
    """Create all required directories for the project."""
    emotion_categories = ["happy", "sad", "angry", "disgust", "neutral", "surprise"]
    play_dir_base_path = "music"

    print("Setting up project directories...\n")

    for emotion in emotion_categories:
        # Training data directory
        os.makedirs(emotion, exist_ok=True)
        print(f"  ✓ {emotion}/")

        # Music playback directory
        music_dir = os.path.join(play_dir_base_path, emotion)
        os.makedirs(music_dir, exist_ok=True)
        print(f"  ✓ {music_dir}/")

    # Directory for songs awaiting classification
    os.makedirs("songs_to_classify", exist_ok=True)
    print(f"  ✓ songs_to_classify/")

    print("\nAll directories created successfully!")
    print("Next steps:")
    print("  1. Place training .mp3 files in emotion folders (happy/, sad/, etc.)")
    print("  2. Run: python train_model.py")


if __name__ == "__main__":
    setup_directories()
