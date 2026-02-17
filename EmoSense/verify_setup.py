import sys
import os

print("=" * 60)
print("EMOSENSE - Setup Verification")
print("=" * 60)

print(f"\nPython version: {sys.version}")
if sys.version_info < (3, 8):
    print("ERROR: Python 3.8+ required!")
    sys.exit(1)

print("\nChecking required packages:")

packages = {
    'tensorflow': 'TensorFlow (CNN model)',
    'torch': 'PyTorch (VAD)',
    'transformers': 'Transformers (Text analysis)',
    'librosa': 'Librosa (Audio processing)',
    'sounddevice': 'SoundDevice (Audio capture)',
    'speech_recognition': 'Speech Recognition',
    'matplotlib': 'Matplotlib (GUI plots)',
    'numpy': 'NumPy',
    'scipy': 'SciPy',
    'sklearn': 'Scikit-learn',
    'resemblyzer': 'Resemblyzer (Speaker recognition)',
    'nltk': 'NLTK',
}

missing = []
for package, description in packages.items():
    try:
        if package == 'sklearn':
            __import__('sklearn')
        else:
            __import__(package)
        print(f"{description}")
    except ImportError:
        print(f" {description} - MISSING!")
        missing.append(package)

if missing:
    print(f"\nMissing packages: {', '.join(missing)}")
    print("  Run: pip install -r requirements.txt")
else:
    print("\nAll packages installed!")

print("\nChecking model files:")
files = [
    ('files/emotion_cnn_plus.keras', 'CNN emotion model'),
    ('files/label_encoder.npy', 'Label encoder'),
    ('files/app_settings.json', 'Settings file (optional)'),
]

missing_files = []
for filename, description in files:
    if os.path.exists(filename):
        size = os.path.getsize(filename) / 1024 / 1024
        print(f"{description} ({size:.1f} MB)")
    else:
        if 'optional' not in description.lower():
            print(f" {description} - NOT FOUND!")
            missing_files.append(filename)
        else:
            print(f"{description} - Not found (will be created)")

if missing_files:
    print(f"\n Missing required files: {', '.join(missing_files)}")
    print("  Please ensure model files are in the project directory")

print("\nChecking audio devices:")
try:
    import sounddevice as sd

    devices = sd.query_devices()
    input_devices = [d for d in devices if d['max_input_channels'] > 0]

    if input_devices:
        print(f"Found {len(input_devices)} input device(s):")
        for i, d in enumerate(input_devices[:3]):
            print(f"{i + 1}. {d['name']} ({d['max_input_channels']}ch)")
        if len(input_devices) > 3:
            print(f"... and {len(input_devices) - 3} more")
    else:
        print("No audio input devices found!")

except Exception as e:
    print(f"Error checking audio devices: {e}")

print("\nChecking disk space:")
try:
    import shutil

    stat = shutil.disk_usage(".")
    free_gb = stat.free / (1024 ** 3)
    print(f" Free space: {free_gb:.1f} GB")
    if free_gb < 2:
        print("Warning: Low disk space!")
except Exception:
    print("Could not check disk space")

print("\n" + "=" * 60)
if not missing and not missing_files:
    print("EMOSENSE READY TO RUN! Execute: python main.py")
else:
    print("SETUP INCOMPLETE - Please fix the issues above")
    if missing:
        print("\n1. Install missing packages:")
        print(f"   pip install {' '.join(missing)}")
    if missing_files:
        print("\n2. Add missing model files to project directory")

print("=" * 60)