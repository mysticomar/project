#!/usr/bin/env python3
"""
Emotion-Based Music Player — Main Application
================================================
Real-time facial emotion detection using DeepFace + OpenCV that
automatically plays music matching the user's detected mood.

Usage:
    python emotional.py              # Use default webcam (ID: 0)
    python emotional.py --cam-id 1   # Use specific webcam

Controls:
    Press 'q' to quit the application.

How it works:
    1. Captures webcam feed frame-by-frame
    2. Analyzes each frame for facial emotion using DeepFace
    3. After detecting a consistent emotion for ~1 second, picks a random
       song from the corresponding music/<emotion>/ folder
    4. Plays the song via mpv media player
    5. Resumes emotion scanning after the song finishes

Supported emotions: happy, sad, angry, surprise, disgust, neutral

Author: Omar Faruk
"""

from deepface import DeepFace
import cv2
import subprocess
import time
import os
import random
import argparse


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Emotion-Based Music Recommendation System'
    )
    parser.add_argument(
        '--cam-id', type=int, default=0,
        help='ID of the webcam to use (default: 0)'
    )
    return parser.parse_args()


def get_emotion(frame):
    """
    Detect the dominant emotion and face region from a video frame.

    Args:
        frame (np.ndarray): BGR image from the webcam.

    Returns:
        tuple: (emotion_label, face_region_dict_or_None)
    """
    try:
        result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=True)
        emotion = result[0]["dominant_emotion"]
        face_region = result[0].get('region', None)
        return emotion, face_region
    except Exception:
        return None, None


def draw_face_box(frame, face_region):
    """Draw a green bounding box around the detected face."""
    if face_region:
        x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def play_song(song_path):
    """
    Launch mpv to play a song in a separate process.

    Args:
        song_path (str): Path to the audio file.

    Returns:
        subprocess.Popen: The mpv process handle.
    """
    return subprocess.Popen([
        "mpv", "--osc",
        "--script-opts=osc-visibility=always",
        "--force-window=yes",
        song_path
    ])


def main():
    """Main application loop."""
    args = parse_args()

    # Configuration
    music_base_path = "music"
    scan_wait_time = 1  # seconds of consistent emotion before playing

    # Initialize webcam
    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        print(f"Error: Could not open webcam (ID: {args.cam_id})")
        return

    # State variables
    current_emotion = "neutral"
    emotion_start_time = time.time()
    is_playing = False
    mpv_process = None
    current_song = ""
    face_detected = False

    cv2.namedWindow("Capture", cv2.WINDOW_GUI_NORMAL)
    print("Emotion-Based Music Player started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # Mirror the frame

        if is_playing:
            # Check if the song has finished playing
            if mpv_process and mpv_process.poll() is not None:
                is_playing = False
                mpv_process = None
                emotion_start_time = time.time()

            # Display current emotion and song name
            song_name = os.path.splitext(current_song)[0]
            text = f"{current_emotion}: {song_name}"
        else:
            # Detect emotion from the current frame
            detected_emotion, face_region = get_emotion(frame)

            if detected_emotion:
                face_detected = True
                draw_face_box(frame, face_region)
            else:
                detected_emotion = current_emotion
                face_detected = False

            # Track emotion consistency
            if detected_emotion != current_emotion:
                current_emotion = detected_emotion
                emotion_start_time = time.time()

            duration = time.time() - emotion_start_time

            # Play music after consistent emotion detection
            if duration > scan_wait_time and face_detected:
                folder = os.path.join(music_base_path, current_emotion)
                if os.path.isdir(folder):
                    songs = [
                        f for f in os.listdir(folder)
                        if f.lower().endswith(('.mp3', '.wav', '.ogg', '.m4a'))
                    ]
                    if songs:
                        current_song = random.choice(songs)
                        song_path = os.path.join(folder, current_song)
                        mpv_process = play_song(song_path)
                        is_playing = True
                        print(f"[{current_emotion}] Playing: {current_song}")
                emotion_start_time = time.time()

            text = current_emotion if face_detected else ""

        # Overlay text on the frame
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
        cv2.imshow("Capture", frame)

        # Quit on 'q' key press
        if cv2.waitKey(30) & 0xFF == ord('q'):
            if mpv_process:
                mpv_process.terminate()
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")


if __name__ == "__main__":
    main()
