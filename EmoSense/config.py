"""
Centralized configuration for EmoSense.

Contains all tunable parameters, model references, shared state,
and runtime caches used across the application.
"""

from queue import Queue
from collections import deque
from concurrent.futures import ThreadPoolExecutor

PYTORCH_AVAILABLE: bool = False
_torch_module = None
try:
    import torch as _torch_ref
    PYTORCH_AVAILABLE = True
    _torch_module = _torch_ref
except ImportError:
    pass


CHUNK_SEC = 1.0
SILERO_VAD_THRESHOLD = 0.6
MAX_SPEECH_SEGMENT_DURATION_S = 7.0
ENERGY_THRESHOLD = 0.015
EMOTION_SENSITIVITY = 2.2
DISTRESS_WINDOW_SECONDS = 30
DISTRESS_THRESHOLD_PERCENTAGE = 0.7
DISTRESS_MIN_CONFIDENCE = 0.4

CNN_PATH = "files/emotion_cnn_plus.keras"
LABEL_PATH = "files/label_encoder.npy"
SETTINGS_FILE = "files/app_settings.json"
PEAK_SR = 22050
VAD_SR = 16000
CNN_EXPECTED_MFCC_FEATURES = 40
CNN_EXPECTED_TIME_STEPS = 173

NEW_SPK_TH = 0.9
CENTROID_MOMENTUM = 0.75
DIA_MAX_INACTIVE_TIME = 120
RESET_TO_NEUTRAL_AFTER = 7.0

CNN_NEUTRALITY_INJECTION = 1.0

EXTERNAL_AUDIO_VERY_HIGH_RMS = 0.12
EXTERNAL_AUDIO_HIGH_RMS = 0.08
EXTERNAL_AUDIO_VERY_LOW_RMS = 0.025
INTERNAL_AUDIO_VERY_HIGH_RMS = 0.7
INTERNAL_AUDIO_HIGH_RMS = 0.45
INTERNAL_AUDIO_VERY_LOW_RMS = 0.15

SPAM_CONFIDENCE_THRESHOLD = 0.5
SPAM_MISMATCH_PAIRS = {
    ("happy", "angry"), ("angry", "happy"),
    ("happy", "sad"), ("sad", "happy")
}

SCROLL_WINDOW_SEC = 10
MAX_GUI_SPK = 2
EMO_COL = {"sad": "#4169E1", "angry": "#FF4500", "happy": "#32CD32", "neutral": "#87CEEB", "__processing__": "#808080", "silent": "#D3D3D3"}

AUDIO_SOURCE_EXTERNAL = "External Microphone"
AUDIO_SOURCE_INTERNAL = "Internal Audio (System Output)"
TARGET_EMOTIONS = ["neutral", "angry", "happy", "sad"]
INITIAL_EMOTION_WEIGHTS = {"cnn": 0.35, "logic": 0.35, "text": 0.30}

cnn_model, cnn_lblenc = None, None
speaker_encoder, vad_model, get_speech_timestamps = None, None, None
gui_speaker_mapper = None
audio_source_var, internal_audio_device_var = None, None
text_emotion_pipeline = None
torch = _torch_module  # Runtime reference, set by main.py during initialization
current_audio_device_index = -1
segments_data = deque()
segment_id_map = {}
stt_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='STT_Thread')
stt_futures = []
emotion_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='Emotion_Thread')
emotion_futures = []
loading_progress_queue = Queue()
_last_significant_emotion_cache_spk = {i: {'emotion': "N/A", 'time': 0} for i in range(MAX_GUI_SPK)}
_current_1s_emotion_cache_spk = {i: {"text": "Analyzing..."} for i in range(MAX_GUI_SPK)}
audio_loop_last_dev_idx_val = -2
audio_loop_stream_obj_ref = [None]
emotion_weight_vars = {}
GUI_UPDATE_INTERVAL = 250
LOADING_MIN_TIME = 4.0

distress_detection_history = {i: deque(maxlen=60) for i in range(MAX_GUI_SPK)}
current_distress_status = {i: {'at_risk': False, 'emotion': None, 'duration': 0, 'confidence': 0}
                          for i in range(MAX_GUI_SPK)}