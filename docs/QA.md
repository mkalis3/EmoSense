# EmoSense QA Notes

## Automated Checks

Run these commands from the repository root:

```powershell
python -m compileall EmoSense
python scripts/check_project.py
python -m pytest EmoSense/tests -q
```

The GitHub Actions workflow runs the same checks on every push and pull request.

## Test Coverage

The current unit tests cover:

- audio feature extraction and cache behavior
- CNN fallback behavior when the model is unavailable
- acoustic emotion scoring
- text emotion scoring for English and Hebrew inputs
- cross-modal risk scoring
- distress monitoring
- speaker diarization and GUI speaker mapping
- report generation
- utility functions

## Manual Verification

Before a demo or interview, run `python verify_setup.py` from the `EmoSense/` package directory. It validates the Python version, required packages, model files, microphone access, and available disk space.

For an application smoke test:

1. Run `python main.py` from the package directory.
2. Confirm the loading screen completes.
3. Select a microphone or internal audio source.
4. Speak for at least ten seconds and confirm waveform segments are drawn.
5. Open a segment and verify the CNN, acoustic, text, and final decision fields.
6. Generate a report and confirm it contains per-speaker summaries.
