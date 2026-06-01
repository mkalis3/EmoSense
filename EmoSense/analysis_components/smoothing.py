"""Emotion smoothing state."""

import time
from collections import deque

import config

emotion_history = {i: deque(maxlen=5) for i in range(config.MAX_GUI_SPK)}

def smooth_emotion_result(speaker_id, current_emotion, current_confidence, segment_id):
    if 0 <= speaker_id < config.MAX_GUI_SPK:
        history = emotion_history[speaker_id]

        history.append({
            'emotion': current_emotion,
            'confidence': current_confidence,
            'timestamp': time.time(),
            'segment_id': segment_id
        })

        if len(history) < 2:
            return current_emotion, current_confidence

        if current_confidence > 0.5:
            return current_emotion, current_confidence

        if current_confidence < 0.35:
            recent = list(history)[-3:]
            emotion_counts = {}

            for entry in recent:
                emo = entry['emotion']
                conf = entry['confidence']
                emotion_counts[emo] = emotion_counts.get(emo, 0) + conf

            if emotion_counts:
                smoothed_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
                avg_confidence = sum(e['confidence'] for e in recent) / len(recent)

                if smoothed_emotion != current_emotion and emotion_counts[smoothed_emotion] > emotion_counts.get(
                        current_emotion, 0) * 1.5:
                    return smoothed_emotion, avg_confidence

        return current_emotion, current_confidence

    return current_emotion, current_confidence
