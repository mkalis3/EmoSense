"""Environment checks for running the EmoSense desktop application."""

from __future__ import annotations

import importlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
MIN_PYTHON = (3, 9)


@dataclass(frozen=True)
class PackageRequirement:
    import_name: str
    display_name: str
    install_name: str | None = None

    @property
    def package_name(self) -> str:
        return self.install_name or self.import_name


REQUIRED_PACKAGES = [
    PackageRequirement("tensorflow", "TensorFlow (CNN model)"),
    PackageRequirement("torch", "PyTorch (VAD)"),
    PackageRequirement("transformers", "Transformers (text analysis)"),
    PackageRequirement("librosa", "Librosa (audio processing)"),
    PackageRequirement("sounddevice", "SoundDevice (audio capture)"),
    PackageRequirement("speech_recognition", "Speech Recognition"),
    PackageRequirement("matplotlib", "Matplotlib (GUI plots)"),
    PackageRequirement("numpy", "NumPy"),
    PackageRequirement("scipy", "SciPy"),
    PackageRequirement("sklearn", "Scikit-learn", "scikit-learn"),
    PackageRequirement("resemblyzer", "Resemblyzer (speaker recognition)"),
    PackageRequirement("nltk", "NLTK"),
]

MODEL_FILES = [
    ("files/emotion_cnn_plus.keras", "CNN emotion model", True),
    ("files/label_encoder.npy", "Label encoder", True),
    ("files/app_settings.json", "Settings file", False),
]


def print_header(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def check_python_version() -> bool:
    print_header("Python")
    print(f"Version: {sys.version.split()[0]}")

    if sys.version_info < MIN_PYTHON:
        print(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is required.")
        return False

    print("Python version is supported.")
    return True


def check_required_packages() -> list[str]:
    print_header("Packages")
    missing = []

    for requirement in REQUIRED_PACKAGES:
        try:
            importlib.import_module(requirement.import_name)
            print(f"OK  {requirement.display_name}")
        except ImportError:
            print(f"Missing  {requirement.display_name}")
            missing.append(requirement.package_name)

    return missing


def check_model_files() -> list[str]:
    print_header("Model Files")
    missing = []

    for relative_path, description, required in MODEL_FILES:
        path = PROJECT_DIR / relative_path
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"OK  {description} ({size_mb:.1f} MB)")
            continue

        if required:
            print(f"Missing  {description}: {path}")
            missing.append(relative_path)
        else:
            print(f"Optional  {description}: will be created when needed")

    return missing


def check_audio_devices() -> None:
    print_header("Audio Devices")
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        input_devices = [device for device in devices if device["max_input_channels"] > 0]
    except Exception as exc:
        print(f"Audio device check skipped: {exc}")
        return

    if not input_devices:
        print("No audio input devices were detected.")
        return

    print(f"Found {len(input_devices)} input device(s).")
    for index, device in enumerate(input_devices[:3], start=1):
        print(f"{index}. {device['name']} ({device['max_input_channels']}ch)")
    if len(input_devices) > 3:
        print(f"... and {len(input_devices) - 3} more")


def check_disk_space() -> bool:
    print_header("Disk Space")
    usage = shutil.disk_usage(PROJECT_DIR)
    free_gb = usage.free / 1024**3
    print(f"Free space: {free_gb:.1f} GB")

    if free_gb < 2:
        print("At least 2 GB of free disk space is recommended.")
        return False

    return True


def main() -> int:
    print("=" * 60)
    print("EMOSENSE - Setup Verification")
    print("=" * 60)

    python_ok = check_python_version()
    missing_packages = check_required_packages()
    missing_files = check_model_files()
    check_audio_devices()
    disk_ok = check_disk_space()

    print("\n" + "=" * 60)
    if python_ok and disk_ok and not missing_packages and not missing_files:
        print("EmoSense is ready to run.")
        print("Run: python main.py")
        print("=" * 60)
        return 0

    print("Setup is incomplete.")
    if missing_packages:
        print("\nInstall missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        print("or run: pip install -r requirements.txt")
    if missing_files:
        print("\nAdd the required model files under:")
        print(PROJECT_DIR / "files")
    print("=" * 60)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
