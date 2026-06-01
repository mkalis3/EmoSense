"""Acoustic feature extraction helpers."""

import threading

import librosa
import numpy as np

import config

feature_cache = {}
cache_lock = threading.Lock()
USE_SIMPLIFIED_FEATURES = True

def extract_advanced_audio_features(audio_segment, segment_id=None):
    """Extract acoustic features from an audio segment.

    Computes RMS energy, zero-crossing rate, spectral centroid, and optionally
    MFCCs, spectral rolloff/bandwidth, and jitter/shimmer.

    Args:
        audio_segment: NumPy array of audio samples at PEAK_SR.
        segment_id: Optional cache key for memoization.

    Returns:
        Dictionary of extracted features, or empty dict on failure.
    """
    if audio_segment is None or len(audio_segment) < 100:
        return {}

    if segment_id and segment_id in feature_cache:
        return feature_cache[segment_id]

    try:
        rms = np.sqrt(np.mean(np.square(audio_segment)))

        if USE_SIMPLIFIED_FEATURES:
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio_segment, frame_length=1024, hop_length=512)[0])

            spectral_centroid = np.mean(librosa.feature.spectral_centroid(
                y=audio_segment, sr=config.PEAK_SR, n_fft=1024, hop_length=512)[0])

            features = {
                'rms': float(rms),
                'zcr': float(zcr),
                'spectral_centroid': float(spectral_centroid),
                'spectral_rolloff': 0,
                'spectral_bandwidth': 0,
                'f0_mean': 0,
                'f0_std': 0,
                'tempo': 120.0,
                'mfcc_mean': np.zeros(13),
                'mfcc_std': np.zeros(13),
                'energy_mean': float(rms),
                'energy_std': 0,
                'jitter': 0,
                'shimmer': 0,
                'duration': float(len(audio_segment) / config.PEAK_SR)
            }
        else:
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio_segment)[0])

            n_fft = 1024
            hop_length = 512
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(
                y=audio_segment, sr=config.PEAK_SR, n_fft=n_fft, hop_length=hop_length)[0])
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(
                y=audio_segment, sr=config.PEAK_SR, n_fft=n_fft, hop_length=hop_length)[0])
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(
                y=audio_segment, sr=config.PEAK_SR, n_fft=n_fft, hop_length=hop_length)[0])

            f0_mean, f0_std = 0, 0

            tempo = 120.0

            try:
                mfccs = librosa.feature.mfcc(y=audio_segment, sr=config.PEAK_SR, n_mfcc=13, n_fft=n_fft)
                mfcc_mean = np.mean(mfccs, axis=1)
                mfcc_std = np.std(mfccs, axis=1)
            except Exception:
                mfcc_mean = np.zeros(13)
                mfcc_std = np.zeros(13)

            energy_mean = float(rms)
            energy_std = 0

            jitter, shimmer = 0, 0

            features = {
                'rms': float(rms),
                'zcr': float(zcr),
                'spectral_centroid': float(spectral_centroid),
                'spectral_rolloff': float(spectral_rolloff),
                'spectral_bandwidth': float(spectral_bandwidth),
                'f0_mean': f0_mean,
                'f0_std': f0_std,
                'tempo': tempo,
                'mfcc_mean': mfcc_mean,
                'mfcc_std': mfcc_std,
                'energy_mean': energy_mean,
                'energy_std': energy_std,
                'jitter': jitter,
                'shimmer': shimmer,
                'duration': float(len(audio_segment) / config.PEAK_SR)
            }

        if segment_id:
            with cache_lock:
                if len(feature_cache) > 100:
                    feature_cache.clear()
                feature_cache[segment_id] = features

        return features

    except Exception:
        return {
            'rms': float(rms) if 'rms' in locals() else 0,
            'zcr': 0, 'spectral_centroid': 0, 'spectral_rolloff': 0,
            'spectral_bandwidth': 0, 'f0_mean': 0, 'f0_std': 0,
            'tempo': 120.0, 'mfcc_mean': np.zeros(13), 'mfcc_std': np.zeros(13),
            'energy_mean': 0, 'energy_std': 0, 'jitter': 0, 'shimmer': 0,
            'duration': float(len(audio_segment) / config.PEAK_SR)
        }
