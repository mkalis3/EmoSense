"""Rule-based acoustic emotion analysis."""

import numpy as np

import config
from analysis_components.audio_features import extract_advanced_audio_features

def logic_based_emotion_analysis(audio_segment, is_internal_audio, segment_id=None):
    """Analyze emotion using acoustic features and rule-based heuristics.

    Uses RMS energy levels, spectral centroid, zero-crossing rate, and
    MFCC variability to estimate emotion distribution.

    Args:
        audio_segment: NumPy array of audio samples.
        is_internal_audio: Whether audio is from system output (different thresholds).
        segment_id: Optional cache key for feature extraction.

    Returns:
        Tuple of (probability_distribution, reason_string).
    """
    if audio_segment is None or len(audio_segment) < 100:
        return {"neutral": 0.55, "happy": 0.15, "sad": 0.15, "angry": 0.15}, "No audio"

    features = extract_advanced_audio_features(audio_segment, segment_id)
    if not features:
        return {"neutral": 0.55, "happy": 0.15, "sad": 0.15, "angry": 0.15}, "No features"

    emotion_scores = {
        "neutral": 0.35,
        "happy": 0.22,
        "sad": 0.22,
        "angry": 0.21
    }

    if is_internal_audio:
        very_high_rms = config.INTERNAL_AUDIO_VERY_HIGH_RMS
        high_rms = config.INTERNAL_AUDIO_HIGH_RMS
        moderate_rms = high_rms * 0.7
        low_rms = config.INTERNAL_AUDIO_VERY_LOW_RMS
        very_low_rms = low_rms * 0.7
    else:
        very_high_rms = config.EXTERNAL_AUDIO_VERY_HIGH_RMS
        high_rms = config.EXTERNAL_AUDIO_HIGH_RMS
        moderate_rms = 0.05
        low_rms = config.EXTERNAL_AUDIO_VERY_LOW_RMS
        very_low_rms = 0.015

    rms = features['rms']

    intense_multiplier = 1.5

    if rms > very_high_rms:
        intensity = min(1.0, (rms - very_high_rms) / very_high_rms)
        emotion_scores["angry"] += 0.15 * intensity * intense_multiplier
        emotion_scores["happy"] += 0.05 * intensity * intense_multiplier
        emotion_scores["neutral"] -= 0.15 * intensity * intense_multiplier

    elif rms > high_rms:
        intensity = (rms - high_rms) / (very_high_rms - high_rms)
        emotion_scores["happy"] += 0.15 * (1 - intensity * 0.3) * intense_multiplier
        emotion_scores["angry"] += 0.05 * intensity * intense_multiplier
        emotion_scores["neutral"] -= 0.1 * intense_multiplier

    elif rms > moderate_rms:
        intensity = (rms - moderate_rms) / (high_rms - moderate_rms)
        emotion_scores["happy"] += 0.08 * intensity * intense_multiplier
        emotion_scores["neutral"] -= 0.05 * intensity * intense_multiplier

    elif rms < very_low_rms:
        intensity = min(1.0, (very_low_rms - rms) / very_low_rms)
        emotion_scores["sad"] += 0.15 * intensity * intense_multiplier
        emotion_scores["neutral"] -= 0.1 * intensity * intense_multiplier

    elif rms < low_rms:
        intensity = (low_rms - rms) / (low_rms - very_low_rms)
        emotion_scores["sad"] += 0.08 * intensity * intense_multiplier
        emotion_scores["neutral"] += 0.05 * (1 - intensity)
    else:
        emotion_scores["neutral"] += 0.1 * 0.5

    centroid = features.get('spectral_centroid', 0)
    spectral_impact = 0.5 if is_internal_audio else 0.7

    if centroid > 4000:
        intensity = min(1.0, (centroid - 4000) / 2000)
        emotion_scores["happy"] += 0.12 * intensity * spectral_impact
        emotion_scores["neutral"] -= 0.08 * intensity * spectral_impact

    elif centroid > 3000:
        intensity = (centroid - 3000) / 1000
        emotion_scores["happy"] += 0.08 * intensity * spectral_impact
        emotion_scores["neutral"] -= 0.04 * intensity * spectral_impact

    elif centroid > 2000:
        if centroid > 2500:
            emotion_scores["happy"] += 0.03 * spectral_impact
        else:
            emotion_scores["neutral"] += 0.03 * spectral_impact

    elif centroid < 1000:
        intensity = min(1.0, (1000 - centroid) / 500)
        emotion_scores["sad"] += 0.12 * intensity * spectral_impact
        emotion_scores["neutral"] -= 0.08 * intensity * spectral_impact

    elif centroid < 1500:
        intensity = (1500 - centroid) / 500
        emotion_scores["sad"] += 0.06 * intensity * spectral_impact
        emotion_scores["angry"] += 0.03 * intensity * spectral_impact
        emotion_scores["neutral"] -= 0.05 * intensity * spectral_impact

    if not is_internal_audio:
        zcr = features.get('zcr', 0)
        if zcr > 0.15:
            emotion_scores["angry"] += 0.06
            emotion_scores["happy"] += 0.03
            emotion_scores["neutral"] -= 0.06
        elif zcr > 0.1:
            emotion_scores["happy"] += 0.04
            emotion_scores["angry"] += 0.02
            emotion_scores["neutral"] -= 0.04
        elif zcr < 0.03:
            emotion_scores["sad"] += 0.03
            emotion_scores["neutral"] += 0.02

    duration = features.get('duration', 1.0)
    if duration > 0.5:
        if 'mfcc_std' in features and len(features['mfcc_std']) > 0:
            variability = np.mean(features['mfcc_std'][:3])
            if variability > 5.0:
                emotion_scores["angry"] += 0.04
                emotion_scores["neutral"] -= 0.03
            elif variability < 1.0:
                emotion_scores["sad"] += 0.03
                emotion_scores["neutral"] += 0.02

    max_emotion_score = 0.65
    for emotion in ['happy', 'sad', 'angry']:
        if emotion_scores[emotion] > max_emotion_score:
            excess = emotion_scores[emotion] - max_emotion_score
            emotion_scores[emotion] = max_emotion_score
            emotion_scores['neutral'] += excess * 0.5

    min_neutral = 0.15
    if emotion_scores['neutral'] < min_neutral:
        deficit = min_neutral - emotion_scores['neutral']
        emotion_scores['neutral'] = min_neutral
        for emotion in ['happy', 'sad', 'angry']:
            emotion_scores[emotion] *= (1 - deficit * 0.3)

    emotion_scores = {k: max(0.05, v) for k, v in emotion_scores.items()}
    total = sum(emotion_scores.values())
    if total > 0:
        emotion_scores = {k: v / total for k, v in emotion_scores.items()}

    reason_parts = []
    audio_type = "Int" if is_internal_audio else "Ext"

    if rms > high_rms:
        reason_parts.append(f"Loud ({rms:.3f})")
    elif rms < low_rms:
        reason_parts.append(f"Quiet ({rms:.3f})")
    else:
        reason_parts.append("Normal vol")

    if centroid > 3000:
        reason_parts.append("Bright")
    elif centroid < 1500:
        reason_parts.append("Dark")
    else:
        reason_parts.append("Balanced")

    max_emotion = max(emotion_scores.items(), key=lambda x: x[1])
    if max_emotion[1] > 0.35 and max_emotion[0] != 'neutral':
        reason_parts.append(f"->{max_emotion[0]}")

    reason = f"Logic[{audio_type}]: {', '.join(reason_parts)}"

    return emotion_scores, reason
