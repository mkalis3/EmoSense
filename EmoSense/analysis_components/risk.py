"""Risk and distress detection."""

import time
from collections import deque

import numpy as np

import config


def analyze_spam_detection(text_dist, voice_dist, text_content=""):
    """Detect potential spam or deceptive speech.

    Checks for cross-modal emotion mismatches, repetitive text patterns,
    overly neutral signals, and unstable emotion patterns.

    Args:
        text_dist: Emotion distribution from text analysis.
        voice_dist: Emotion distribution from voice analysis.
        text_content: Raw transcribed text.

    Returns:
        Tuple of (status_dict, reason_string).
    """
    if not text_dist or not voice_dist:
        return {"status": "Not Detected", "confidence": 0.05}, "Not enough data"

    spam_score = 0.0
    reasons = []

    text_emos = {k: v for k, v in text_dist.items() if k != 'neutral'}
    voice_emos = {k: v for k, v in voice_dist.items() if k != 'neutral'}

    if text_emos and voice_emos:
        text_top_emo, text_top_score = max(text_emos.items(), key=lambda x: x[1])
        voice_top_emo, voice_top_score = max(voice_emos.items(), key=lambda x: x[1])

        extreme_mismatches = {
            ("happy", "angry"), ("angry", "happy"),
            ("happy", "sad"), ("sad", "happy")
        }

        if (text_top_emo, voice_top_emo) in extreme_mismatches and text_top_score > 0.4 and voice_top_score > 0.4:
            spam_score += 0.3
            reasons.append(f"Emotion mismatch: text '{text_top_emo}' vs voice '{voice_top_emo}'")

    if text_content:
        words = text_content.lower().split()
        if len(words) > 5:
            unique_words = len(set(words))
            repetition_ratio = unique_words / len(words)
            if repetition_ratio < 0.5:
                spam_score += 0.2
                reasons.append("Repetitive text pattern")

    if text_dist.get('neutral', 0) > 0.8 or voice_dist.get('neutral', 0) > 0.8:
        spam_score += 0.15
        reasons.append("Overly neutral emotion")

    emotion_variance = np.var([text_dist.get(e, 0) - voice_dist.get(e, 0) for e in ['happy', 'sad', 'angry']])
    if emotion_variance > 0.2:
        spam_score += 0.1
        reasons.append("Unstable emotion pattern")

    confidence = min(0.95, max(0.05, spam_score))

    if confidence > 0.5:
        status = "Spam Detected"
    else:
        status = "Not Detected"

    reason_text = "; ".join(reasons) if reasons else "Normal conversation pattern"

    return {"status": status, "confidence": confidence}, reason_text


def check_distress_detection(speaker_id, emotion, confidence, text_content=""):
    """Monitor for prolonged negative emotional states.

    Tracks emotion history in a sliding window and triggers distress alerts
    when negative emotions exceed configurable thresholds.

    Args:
        speaker_id: Speaker GUI slot index.
        emotion: Detected emotion label.
        confidence: Detection confidence (0-1).
        text_content: Raw text for fear-word detection.
    """
    if speaker_id < 0 or speaker_id >= config.MAX_GUI_SPK:
        return

    if not hasattr(config, 'distress_detection_history'):
        config.distress_detection_history = {i: deque(maxlen=60) for i in range(config.MAX_GUI_SPK)}

    if not hasattr(config, 'current_distress_status'):
        config.current_distress_status = {i: {'at_risk': False, 'emotion': None, 'duration': 0, 'confidence': 0}
                                          for i in range(config.MAX_GUI_SPK)}

    FEAR_WORDS = {
        'scared', 'terrified', 'terrifying', 'afraid', 'frightened', 'fear', 'fearful',
        'panic', 'panicking', 'horror', 'horrified', 'dread', 'dreading', 'anxious',
        'petrified', 'alarmed', 'shocked', 'nightmare', 'threatening', 'danger',
        'מפחד', 'מפחדת', 'פחד', 'מבוהל', 'מבוהלת', 'חושש', 'חוששת', 'נבהל', 'נבהלת',
        'מפוחד', 'מפוחדת', 'אימה', 'זוועה', 'סיוט', 'מאיים', 'סכנה'
    }

    has_fear = False
    if text_content:
        text_lower = text_content.lower()
        has_fear = any(fear_word in text_lower for fear_word in FEAR_WORDS)

    if has_fear and emotion in ['sad', 'angry']:
        emotion = 'distress_fear'
        confidence = max(0.7, confidence)

    current_time = time.time()
    config.distress_detection_history[speaker_id].append({
        'emotion': emotion,
        'confidence': confidence,
        'timestamp': current_time,
        'has_fear': has_fear
    })

    history = list(config.distress_detection_history[speaker_id])
    if len(history) < 10:
        return

    distress_count = 0
    fear_count = 0
    total_distress_confidence = 0

    cutoff_time = current_time - 30

    for entry in history:
        if entry['timestamp'] > cutoff_time:
            if entry['emotion'] in ['angry', 'sad', 'distress_fear'] and entry['confidence'] > 0.4:
                distress_count += 1
                total_distress_confidence += entry['confidence']
                if entry.get('has_fear', False):
                    fear_count += 1

    recent_entries = sum(1 for e in history if e['timestamp'] > cutoff_time)
    if recent_entries > 0:
        distress_percentage = distress_count / recent_entries
        avg_confidence = total_distress_confidence / max(1, distress_count)

        threshold = 0.6 if fear_count > 0 else 0.7

        if distress_percentage > threshold and avg_confidence > 0.5:
            emotion_counts = {'angry': 0, 'sad': 0, 'fear': fear_count}
            for entry in history:
                if entry['timestamp'] > cutoff_time and entry['emotion'] in emotion_counts:
                    emotion_counts[entry['emotion']] += 1

            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
            if dominant_emotion == 'fear' or fear_count > 2:
                dominant_emotion = 'fear/anxiety'

            config.current_distress_status[speaker_id] = {
                'at_risk': True,
                'emotion': dominant_emotion,
                'duration': current_time - history[0]['timestamp'],
                'confidence': avg_confidence
            }
        else:
            config.current_distress_status[speaker_id] = {
                'at_risk': False,
                'emotion': None,
                'duration': 0,
                'confidence': 0
            }
