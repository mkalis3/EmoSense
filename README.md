# EmoSense

Real-time speech emotion analysis that combines audio classification, acoustic feature analysis, speaker diarization, and text sentiment into a single desktop application.

The application records live audio, groups speech by speaker, transcribes utterances, and combines three independent emotion signals:

- CNN classification over MFCC audio features
- Rule-based acoustic analysis for energy, zero-crossing rate, and spectral shape
- Text sentiment analysis for English and Hebrew transcripts

## Highlights

- Real-time audio capture with voice activity detection
- Online speaker diarization using voice embeddings
- Weighted emotion fusion across audio, acoustic, and text signals
- Distress monitoring over a rolling time window
- Session reports with per-speaker emotion summaries
- Unit tests for core analysis, configuration, diarization, reports, and utilities

## Repository Layout

```text
EmoSense/
├── EmoSense/
│   ├── analysis.py
│   ├── audio_processing.py
│   ├── config.py
│   ├── diarization.py
│   ├── gui.py
│   ├── main.py
│   ├── report_generator.py
│   ├── training_main.py
│   ├── utils.py
│   ├── files/
│   └── tests/
├── requirements-dev.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/mkalis3/EmoSense.git
cd EmoSense/EmoSense

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python verify_setup.py
python main.py
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

## Tests

From the repository root:

```bash
pip install -r requirements-dev.txt
python -m compileall EmoSense
python -m pytest EmoSense/tests -q
```

## Notes

The trained model files are included for reproducible local runs. For a production deployment, these files should be moved to a release artifact or model registry and loaded during setup.

