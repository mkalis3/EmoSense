"""Public emotion analysis API for EmoSense."""

from analysis_components.acoustic import logic_based_emotion_analysis
from analysis_components.audio_features import extract_advanced_audio_features, feature_cache
from analysis_components.cnn import cnn_emotion_analysis
from analysis_components.pipeline import perform_full_emotion_analysis
from analysis_components.risk import analyze_spam_detection, check_distress_detection
from analysis_components.smoothing import emotion_history, smooth_emotion_result
from analysis_components.text_emotion import text_based_emotion_analysis

__all__ = [
    "analyze_spam_detection",
    "check_distress_detection",
    "cnn_emotion_analysis",
    "emotion_history",
    "extract_advanced_audio_features",
    "feature_cache",
    "logic_based_emotion_analysis",
    "perform_full_emotion_analysis",
    "smooth_emotion_result",
    "text_based_emotion_analysis",
]
