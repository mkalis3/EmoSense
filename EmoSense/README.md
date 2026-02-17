# EmoSense

Real-time multi-modal emotion analysis system that combines CNN-based audio classification, acoustic feature analysis, and NLP text sentiment to detect emotions from speech in real-time.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-VAD-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

EmoSense captures live audio, identifies speakers using voice embeddings, transcribes speech via Google STT, and classifies emotions through a weighted ensemble of three independent models:

- **CNN Model** — Trained on CREMA-D, TESS, and RAVDESS datasets using MFCC features
- **Logic-Based Analyzer** — Rule-based system using acoustic features (RMS energy, spectral centroid, zero-crossing rate)
- **Text Sentiment Analyzer** — NLP pipeline using RoBERTa (`cardiffnlp/twitter-roberta-base-emotion`) with Hebrew support via HeBERT

### Key Features

- **Real-time speaker diarization** using cosine similarity on voice embeddings (Resemblyzer)
- **Voice Activity Detection** using Silero VAD (PyTorch)
- **Bilingual support** — English and Hebrew speech-to-text and sentiment analysis
- **Spam/deception detection** — Cross-modal emotion mismatch analysis
- **Distress monitoring** — Sliding-window alert system for prolonged negative emotional states
- **Configurable model weights** — Adjust CNN/Logic/Text contribution via GUI sliders
- **Session reports** — Exportable emotion analysis summaries per speaker

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐
│ Audio Input  │───▶│ VAD (Silero) │───▶│ Speaker Diarization │
│ (sounddevice)│    │              │    │ (Resemblyzer)       │
└─────────────┘    └──────────────┘    └──────────┬──────────┘
                                                   │
                          ┌────────────────────────┼────────────────────────┐
                          ▼                        ▼                        ▼
                   ┌─────────────┐         ┌─────────────┐         ┌──────────────┐
                   │ CNN Model   │         │ Logic-Based │         │ Google STT   │
                   │ (MFCC→Keras)│         │ (Acoustics) │         │ → NLP Model  │
                   └──────┬──────┘         └──────┬──────┘         └──────┬───────┘
                          │                        │                       │
                          ▼                        ▼                       ▼
                   ┌──────────────────────────────────────────────────────────┐
                   │              Weighted Ensemble Fusion                     │
                   │         + Sensitivity Adjustment + Smoothing             │
                   └──────────────────────┬───────────────────────────────────┘
                                          │
                          ┌───────────────┼───────────────┐
                          ▼               ▼               ▼
                   ┌────────────┐  ┌────────────┐  ┌────────────┐
                   │ Spam       │  │ Distress   │  │ GUI        │
                   │ Detection  │  │ Monitoring │  │ Waveform   │
                   └────────────┘  └────────────┘  └────────────┘
```

## Installation

### Prerequisites

- Python 3.9+
- Working microphone
- ~2GB disk space for models

### Setup

```bash
git clone https://github.com/yourusername/EmoSense.git
cd EmoSense

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

python verify_setup.py  # Verify all dependencies
python main.py          # Launch the application
```

### Model Training (Optional)

To retrain the CNN model on CREMA-D, TESS, or RAVDESS datasets:

```bash
python training_main.py
```

Update the dataset paths in `training_main.py` to match your local setup.

## Usage

1. **Launch** — Run `python main.py`. A loading screen shows model initialization progress.
2. **Select Audio Source** — Choose between external microphone or internal audio.
3. **Adjust Weights** — Use the sliders to set the contribution of each model (CNN / Logic / Text).
4. **Monitor** — The waveform display shows real-time emotion color-coding per speaker.
5. **Click segments** — Click on any waveform segment to view detailed analysis breakdown.
6. **Export** — Click "Generate Report" to save a session summary.

## Configuration

All tunable parameters are centralized in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `EMOTION_SENSITIVITY` | 2.2 | Amplification factor for non-neutral emotions |
| `SILERO_VAD_THRESHOLD` | 0.6 | Voice activity detection confidence threshold |
| `DISTRESS_WINDOW_SECONDS` | 30 | Sliding window for distress monitoring |
| `NEW_SPK_TH` | 0.9 | Cosine similarity threshold for new speaker detection |
| `CNN_NEUTRALITY_INJECTION` | 1.0 | Bias toward neutral in CNN predictions |

## Testing

```bash
pip install pytest
pytest tests/ -v
```

## Project Structure

```
EmoSense/
├── main.py                 # Application entry point
├── config.py               # Centralized configuration
├── analysis.py             # Emotion analysis pipeline (CNN, Logic, Text, Fusion)
├── audio_processing.py     # Audio capture, VAD, STT, processing loop
├── diarization.py          # Speaker identification and tracking
├── gui.py                  # Tkinter GUI with matplotlib waveform display
├── report_generator.py     # Session report export
├── training_main.py        # CNN model training scripts
├── utils.py                # Utility functions
├── verify_setup.py         # Dependency verification
├── tests/
│   ├── test_analysis.py    # Emotion analysis unit tests
│   ├── test_config.py      # Configuration validation tests
│   ├── test_diarization.py # Diarization unit tests
│   ├── test_report_generator.py
│   └── test_utils.py       # Utility function tests
├── files/                  # Pre-trained models and settings
│   ├── emotion_cnn_plus.keras
│   ├── label_encoder.npy
│   └── app_settings.json
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Technology Stack

- **Audio Capture**: sounddevice, soundfile
- **Audio Processing**: librosa, scipy
- **Speech-to-Text**: Google Speech Recognition API
- **Voice Embeddings**: Resemblyzer
- **VAD**: Silero VAD (PyTorch)
- **CNN Training**: TensorFlow / Keras
- **NLP**: HuggingFace Transformers (RoBERTa, HeBERT)
- **GUI**: Tkinter + Matplotlib

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
