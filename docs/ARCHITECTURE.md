# EmoSense Architecture

EmoSense is a desktop application for real-time speech emotion analysis. The runtime pipeline is intentionally split into small modules so each stage can be tested and explained independently.

## Runtime Flow

1. `main.py` initializes the model dependencies and starts the Tkinter application.
2. `audio_processing.py` captures audio, applies voice activity detection, transcribes speech, and submits emotion jobs.
3. `diarization.py` maps voice embeddings to stable speaker slots.
4. `analysis.py` exposes the public analysis API.
5. `analysis_components/` contains the implementation for feature extraction, CNN inference, acoustic scoring, text scoring, fusion, smoothing, and risk monitoring.
6. `gui.py` renders the live waveform, speaker state, weight controls, and report actions.
7. `report_generator.py` exports a session summary from the accumulated segment data.

## Analysis Components

- `audio_features.py`: extracts RMS, zero-crossing rate, centroid, bandwidth, rolloff, pitch, and duration features.
- `cnn.py`: prepares MFCC input and reads emotion probabilities from the trained Keras model.
- `acoustic.py`: scores emotion from audio features when model confidence is unavailable or weak.
- `text_emotion.py`: combines keyword detection with transformer-based sentiment models when they are loaded.
- `pipeline.py`: normalizes model weights, fuses the distributions, applies sensitivity, and returns the final result.
- `risk.py`: tracks cross-modal mismatches and prolonged negative states.
- `smoothing.py`: keeps short emotion history per speaker to reduce visual jitter.

## Model Files

The repository includes the trained model files required for local demonstration. In production, these files should be released through a model registry or release artifact and loaded during setup.

## Training Data

The optional training UI reads datasets from environment variables:

```powershell
$env:EMOSENSE_CREMA_PATH="C:\datasets\CREMA-D"
$env:EMOSENSE_TESS_PATH="C:\datasets\TESS"
$env:EMOSENSE_RAVDESS_PATH="C:\datasets\RAVDESS"
python training_main.py
```

When these variables are not set, the trainer looks under `datasets/` inside the working directory.
