"""Full emotion analysis pipeline."""

import re

import config
from analysis_components.acoustic import logic_based_emotion_analysis
from analysis_components.cnn import cnn_emotion_analysis
from analysis_components.risk import analyze_spam_detection, check_distress_detection
from analysis_components.smoothing import smooth_emotion_result
from analysis_components.text_emotion import text_based_emotion_analysis


def perform_full_emotion_analysis(audio_segment, text_segment, weights, is_internal_audio, speaker_id=None,
                                  segment_id=None):
    """Run the complete emotion analysis pipeline.

    Combines CNN, logic-based, and text-based analyses using weighted
    ensemble fusion, applies sensitivity adjustment, emotion smoothing,
    spam detection, and distress monitoring.

    Args:
        audio_segment: Audio samples at PEAK_SR.
        text_segment: Transcribed text.
        weights: Dict with 'cnn', 'logic', 'text' weight values.
        is_internal_audio: Whether source is system audio.
        speaker_id: Optional speaker GUI slot.
        segment_id: Optional segment identifier.

    Returns:
        Tuple of (emotion_label, details_dict, confidence).
    """
    details = {'raw_text': text_segment or "N/A"}

    cnn_dist, cnn_reason = cnn_emotion_analysis(audio_segment, segment_id)
    details['cnn_analysis'] = {'dist': cnn_dist, 'reason': cnn_reason}

    logic_dist, logic_reason = logic_based_emotion_analysis(audio_segment, is_internal_audio, segment_id)
    details['logic_analysis'] = {'dist': logic_dist, 'reason': logic_reason}

    text_dist, text_reason = text_based_emotion_analysis(text_segment)
    details['text_analysis'] = {'dist': text_dist, 'reason': text_reason}

    total_weight = sum(weights.values())
    norm_weights = {k: v / total_weight for k, v in weights.items()} if total_weight > 0 else {'cnn': 0.33,
                                                                                               'logic': 0.34,
                                                                                               'text': 0.33}
    details['final_weights'] = norm_weights

    final_scores = {emo: 0.0 for emo in config.TARGET_EMOTIONS}
    for emo in config.TARGET_EMOTIONS:
        final_scores[emo] = (
                cnn_dist.get(emo, 0) * norm_weights.get('cnn', 0) +
                logic_dist.get(emo, 0) * norm_weights.get('logic', 0) +
                text_dist.get(emo, 0) * norm_weights.get('text', 0)
        )

    original_scores = final_scores.copy()

    suggested_emotion_match = re.search(r"suggests '(\w+)'", text_reason)
    if suggested_emotion_match:
        suggested_emotion = suggested_emotion_match.group(1).lower()
        if suggested_emotion == 'joy' or suggested_emotion == 'optimism':
            suggested_emotion = 'happy'

        if suggested_emotion in config.TARGET_EMOTIONS and suggested_emotion != 'neutral':
            final_scores[suggested_emotion] += 0.3


    sensitivity = config.EMOTION_SENSITIVITY

    before_sensitivity = sorted([(e, s) for e, s in final_scores.items() if e != 'neutral'],
                                key=lambda x: x[1], reverse=True)

    max_score = max(final_scores.values())
    max_emotion = max(final_scores.items(), key=lambda x: x[1])[0]

    if max_emotion != 'neutral' and max_score > final_scores['neutral']:
        neutral_score = final_scores['neutral']

        for emo in ['happy', 'sad', 'angry']:
            if final_scores[emo] > neutral_score:
                gap = final_scores[emo] - neutral_score
                new_gap = gap * sensitivity
                final_scores[emo] = neutral_score + new_gap
    else:
        final_scores['neutral'] *= 0.9
        boost = (1 - 0.9) * final_scores['neutral'] / 3
        for emo in ['happy', 'sad', 'angry']:
            final_scores[emo] += boost

    after_sensitivity = sorted([(e, s) for e, s in final_scores.items() if e != 'neutral'],
                               key=lambda x: x[1], reverse=True)

    if (before_sensitivity and after_sensitivity and
            before_sensitivity[0][0] != after_sensitivity[0][0] and
            text_dist and text_segment):
        text_emotions = {k: v for k, v in text_dist.items() if k != 'neutral'}
        if text_emotions:
            text_top_emo, text_top_score = max(text_emotions.items(), key=lambda x: x[1])
            if text_top_emo == before_sensitivity[0][0] and text_top_score > 0.4:
                for emo in ['happy', 'sad', 'angry']:
                    final_scores[emo] = original_scores[emo] * 1.2

    total_score = sum(final_scores.values())
    if total_score > 0:
        final_scores = {k: v / total_score for k, v in final_scores.items()}

    min_neutral_floor = 0.15
    if final_scores['neutral'] < min_neutral_floor:
        deficit = min_neutral_floor - final_scores['neutral']
        final_scores['neutral'] = min_neutral_floor

        emotional_total = sum(final_scores.get(emo, 0) for emo in ['happy', 'sad', 'angry'])
        if emotional_total > 0:
            for emo in ['happy', 'sad', 'angry']:
                final_scores[emo] -= deficit * (final_scores[emo] / emotional_total)

    total = sum(final_scores.values())
    if total > 0:
        final_scores = {k: v / total for k, v in final_scores.items()}
    else:
        final_scores = {"neutral": 0.7, "happy": 0.1, "sad": 0.1, "angry": 0.1}

    final_emo, confidence = max(final_scores.items(), key=lambda item: item[1])

    if speaker_id is not None:
        smoothed_emo, smoothed_conf = smooth_emotion_result(speaker_id, final_emo, confidence, segment_id)
        if smoothed_conf > confidence:
            final_emo = smoothed_emo
            confidence = smoothed_conf
            details['smoothing_applied'] = True

    spam_info, spam_reason = analyze_spam_detection(text_dist, logic_dist, text_segment)
    details['spam_detection'] = {
        'status': spam_info['status'],
        'confidence': spam_info['confidence'],
        'reason': spam_reason
    }

    if speaker_id is not None:
        check_distress_detection(speaker_id, final_emo, confidence, text_segment)

    details.update({
        'final_decision': final_emo, 'final_confidence': confidence,
        'final_scores': final_scores.copy()
    })

    return final_emo, details, confidence
