<p align="center">
  <h1 align="center">🎵 Emotion-Based Music Recommendation System</h1>
  <p align="center">
    <em>Real-time facial emotion detection that automatically plays music matching your mood</em>
  </p>
  <p align="center">
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
    <a href="https://opencv.org/"><img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"></a>
    <a href="https://github.com/serengil/deepface"><img src="https://img.shields.io/badge/DeepFace-Emotion_AI-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="DeepFace"></a>
    <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/Scikit--Learn-SVM-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"></a>
    <a href="https://librosa.org/"><img src="https://img.shields.io/badge/Librosa-Audio_Analysis-8B5CF6?style=for-the-badge" alt="Librosa"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge" alt="MIT License"></a>
  </p>
</p>

---

## 📖 Overview

An end-to-end system that detects a user's **facial emotion in real time** via webcam and automatically plays music that matches their current mood. The system combines **computer vision** (DeepFace + OpenCV) for emotion recognition with a **machine learning pipeline** (SVM) for music classification based on audio features.

Built as a **Final Year Major Project (B.Tech CSE, 2025–2026)** at The Assam Royal Global University, Guwahati.

> **Team**: 4 members · **Supervisor**: Ms. Bidisha Goswami

---

## 🏗️ System Architecture

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│   📷 Webcam Feed    │────▶│  🧠 DeepFace         │────▶│  🎭 Detected        │
│   (OpenCV)          │     │  Emotion Analysis     │     │  Emotion Label      │
└─────────────────────┘     └──────────────────────┘     └─────────┬───────────┘
                                                                    │
                                                                    ▼
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│   🔊 Play Song      │◀────│  🎵 Select Random    │◀────│  📁 Emotion-Mapped  │
│   (mpv Player)      │     │  Song from Folder     │     │  Music Library      │
└─────────────────────┘     └──────────────────────┘     └─────────────────────┘
```

### Music Classification Pipeline (Offline)

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│   🎵 MP3 Files      │────▶│  🎼 Feature          │────▶│  📊 MFCC, Tempo,    │
│   (Training Data)   │     │  Extraction (Librosa) │     │  Centroid, ZCR      │
└─────────────────────┘     └──────────────────────┘     └─────────┬───────────┘
                                                                    │
                                                                    ▼
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│   💾 Saved Model    │◀────│  🤖 SVM Classifier   │◀────│  🏋️ Train/Test     │
│   (.pkl files)      │     │  (Linear Kernel)      │     │  Split (80/20)      │
└─────────────────────┘     └──────────────────────┘     └─────────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎭 **Real-Time Emotion Detection** | Detects 6 emotions (happy, sad, angry, surprise, disgust, neutral) using DeepFace |
| 🎵 **Automatic Music Playback** | Plays a random song matching the detected emotion via `mpv` player |
| 🧠 **SVM Music Classifier** | Classifies songs into emotion categories using audio features |
| 🎼 **Audio Feature Extraction** | Extracts 16-dimensional features: 13 MFCCs + tempo + spectral centroid + ZCR |
| 📊 **Confusion Matrix Visualization** | Generates publication-ready confusion matrix heatmaps |
| 🔄 **Auto-Classification** | Batch classifies new songs and sorts them into emotion folders |
| ⏱️ **Emotion Stabilization** | Waits for consistent emotion before switching songs (prevents rapid switching) |

---

## 🗂️ Project Structure

```
project/
├── emotional.py              # 🎯 Main app — webcam emotion detection + music playback
├── feature_extract.py        # 🎼 Audio feature extraction (MFCC, tempo, centroid, ZCR)
├── train_model.py            # 🏋️ SVM model training pipeline
├── confusion_matrix.py       # 📊 Model evaluation & confusion matrix visualization
├── classify_new_mp3.py       # 🔄 Batch classify & sort new songs into emotion folders
├── mkdirs.py                 # 📁 Directory setup script
├── requirements.txt          # 📦 Python dependencies
├── LICENSE                   # ⚖️ MIT License
│
├── music/                    # 🎵 Emotion-sorted music library (for playback)
│   ├── happy/
│   ├── sad/
│   ├── angry/
│   ├── surprise/
│   ├── disgust/
│   └── neutral/
│
├── happy/                    # 🎓 Training data directories
├── sad/                      #    (place training .mp3 files here)
├── angry/
├── surprise/
├── disgust/
├── neutral/
│
├── songs_to_classify/        # 📥 Drop new songs here for auto-classification
│
├── music_classifier_svm.pkl              # 💾 Trained SVM model
├── music_classifier_svm_label_encoder.pkl # 💾 Label encoder
└── test_set.pkl                           # 💾 Saved test set
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **mpv** media player (for music playback)
- **Webcam** (built-in or USB)

### 1. Clone the Repository

```bash
git clone https://github.com/mysticomar/project.git
cd project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install mpv Player

```bash
# macOS
brew install mpv

# Ubuntu/Debian
sudo apt install mpv

# Windows (via Chocolatey)
choco install mpv
```

### 4. Set Up Directory Structure

```bash
python mkdirs.py
```

This creates:
- 6 training directories (`happy/`, `sad/`, `angry/`, `surprise/`, `disgust/`, `neutral/`)
- 6 playback directories inside `music/`
- `songs_to_classify/` for batch classification

### 5. Add Training Data

Place `.mp3` files into the emotion-labeled training directories:
```
happy/       → happy songs
sad/         → sad songs
angry/       → angry songs
surprise/    → surprise songs
disgust/     → dark/intense songs
neutral/     → calm/ambient songs
```

### 6. Train the Model

```bash
python train_model.py
```

This will:
- Extract audio features from all training songs
- Train an SVM classifier (linear kernel, 80/20 split)
- Save the model, label encoder, and test set as `.pkl` files
- Print the training accuracy

### 7. Run the Application

```bash
python emotional.py
```

- The webcam feed opens in a window
- Your detected emotion is shown on screen
- After ~1 second of consistent emotion, a matching song plays
- Press **`q`** to quit

#### Optional: Use a Different Camera

```bash
python emotional.py --cam-id 1
```

---

## 📊 Model Evaluation

Generate a confusion matrix heatmap:

```bash
python confusion_matrix.py
```

This saves `confusion_matrix.png` with a Seaborn heatmap showing classification performance.

---

## 🔄 Classify New Songs

Drop new `.mp3` files into `songs_to_classify/`, then run:

```bash
python classify_new_mp3.py
```

Each song is classified and automatically copied to the matching `music/<emotion>/` folder.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Emotion Detection | [DeepFace](https://github.com/serengil/deepface) + [OpenCV](https://opencv.org/) |
| Audio Feature Extraction | [Librosa](https://librosa.org/) (MFCC, Tempo, Spectral Centroid, ZCR) |
| Music Classification | [Scikit-learn](https://scikit-learn.org/) — SVM (Linear Kernel) |
| Music Playback | [mpv](https://mpv.io/) |
| Visualization | [Matplotlib](https://matplotlib.org/) + [Seaborn](https://seaborn.pydata.org/) |
| Model Persistence | [Joblib](https://joblib.readthedocs.io/) |

---

## 🎼 Audio Features Explained

The system extracts a **16-dimensional feature vector** from each song:

| Feature | Count | Description |
|---------|-------|-------------|
| **MFCC** | 13 | Mel-Frequency Cepstral Coefficients — captures timbral texture |
| **Tempo** | 1 | Beats per minute — captures rhythmic energy |
| **Spectral Centroid** | 1 | "Brightness" of sound — higher = brighter |
| **Zero-Crossing Rate** | 1 | Rate of signal sign changes — captures noisiness/percussiveness |

---

## 👨‍💻 Author

**Omar Faruk**
- 🎓 B.Tech, Computer Science & Engineering — The Assam Royal Global University, Guwahati
- 📧 [mysticomar1289@gmail.com](mailto:mysticomar1289@gmail.com)
- 🔗 [LinkedIn](https://www.linkedin.com/in/omar-faruk-415b643b7)
- 🐙 [GitHub](https://github.com/mysticomar)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with ❤️ as a Final Year Major Project (2025–2026)</sub>
</p>
