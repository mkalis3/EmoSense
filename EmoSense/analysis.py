"""
Emotion analysis pipeline for EmoSense.

Provides three independent analysis methods (CNN, Logic-based, Text-based),
a weighted ensemble fusion, spam detection, distress monitoring, and
emotion smoothing.
"""

import re
import time
import threading
from collections import deque

import numpy as np
import librosa

import config
from utils import contains_predominantly_hebrew

try:
    import scipy.stats
    _ = scipy.stats  # verify availability

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

emotion_history = {i: deque(maxlen=5) for i in range(config.MAX_GUI_SPK)}

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


def cnn_emotion_analysis(audio_segment, segment_id=None):
    """Run CNN model inference on an audio segment.

    Extracts MFCCs, pads/truncates to expected dimensions, runs prediction,
    and applies neutrality injection and anger dampening.

    Args:
        audio_segment: NumPy array of audio samples.
        segment_id: Optional identifier for logging.

    Returns:
        Tuple of (probability_distribution, reason_string).
    """
    if not config.cnn_model or not config.cnn_lblenc or audio_segment is None:
        return {}, "N/A"
    try:
        mfccs = librosa.feature.mfcc(y=audio_segment, sr=config.PEAK_SR, n_mfcc=config.CNN_EXPECTED_MFCC_FEATURES)
        if mfccs.shape[1] > config.CNN_EXPECTED_TIME_STEPS:
            mfccs = mfccs[:, :config.CNN_EXPECTED_TIME_STEPS]
        else:
            mfccs = np.pad(mfccs, ((0, 0), (0, config.CNN_EXPECTED_TIME_STEPS - mfccs.shape[1])), 'constant')

        raw_probs_array = config.cnn_model.predict(mfccs[np.newaxis, ..., np.newaxis], verbose=0)[0]
        raw_dist = {config.cnn_lblenc.classes_[i].lower(): prob for i, prob in enumerate(raw_probs_array)}

        injected_dist = raw_dist.copy()

        injected_dist['neutral'] = injected_dist.get('neutral', 0) + config.CNN_NEUTRALITY_INJECTION

        if 'angry' in injected_dist and injected_dist['angry'] > 0.3:
            anger_reduction = (injected_dist['angry'] - 0.3) * 0.5
            injected_dist['angry'] -= anger_reduction
            injected_dist['neutral'] += anger_reduction * 0.7
            injected_dist['happy'] += anger_reduction * 0.15
            injected_dist['sad'] += anger_reduction * 0.15

        total_prob = sum(injected_dist.values())
        final_dist = {k: v / total_prob for k, v in injected_dist.items()} if total_prob > 0 else {}

        top_emotion = max(final_dist, key=final_dist.get, default="N/A")
        return final_dist, f"Suggests '{top_emotion}' ({final_dist.get(top_emotion, 0) * 100:.0f}%)"
    except Exception as e:
        return {}, f"CNN Error: {e}"


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
        reason_parts.append(f"→{max_emotion[0]}")

    reason = f"Logic[{audio_type}]: {', '.join(reason_parts)}"

    return emotion_scores, reason


def text_based_emotion_analysis(text_input):
    """Analyze emotion from transcribed text.

    Uses keyword matching, laughter detection, RoBERTa NLP pipeline,
    and Hebrew sentiment analysis (HeBERT) to classify emotions.

    Args:
        text_input: Transcribed speech text, or None.

    Returns:
        Tuple of (probability_distribution, reason_string).
    """
    if not text_input or not text_input.strip():
        return {"neutral": 0.6, "happy": 0.13, "sad": 0.13, "angry": 0.14}, "No text"

    is_hebrew = contains_predominantly_hebrew(text_input)

    max_emotion_score = 0
    dominant_emotion = None
    reason = None

    if is_hebrew and not hasattr(config, 'hebrew_sentiment_pipeline'):
        try:
            from transformers import pipeline
            config.hebrew_sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="avichr/heBERT_sentiment_analysis",
                device=-1
            )
        except Exception:
            config.hebrew_sentiment_pipeline = None

    HAPPY_WORDS = {
        'haha', 'hahaha', 'hahahaha', 'lol', 'lmao', 'rofl', 'hehe', 'hihi', 'heehee', 'teehee',
        ':)', ':-)', ':d', ':D', '^_^', 'xD',
        'happy', 'joy', 'joyful', 'glad', 'pleased', 'delighted', 'cheerful', 'excited', 'thrilled',
        'wonderful', 'amazing', 'fantastic', 'great', 'awesome', 'excellent', 'perfect', 'beautiful',
        'love', 'adore', 'enjoy', 'fun', 'funny', 'hilarious', 'smile', 'smiling', 'laugh', 'laughing',
        'yay', 'hooray', 'woohoo', 'yes', 'yeah', 'yup', 'absolutely', 'definitely', 'sure', 'of course',
        'thanks', 'thank you', 'appreciate', 'grateful', 'blessed', 'fortunate', 'lucky',
        'congratulations', 'congrats', 'celebrate', 'party', 'success', 'win', 'winner', 'victory',
        'good', 'nice', 'cool', 'sweet', 'lovely', 'adorable', 'cute', 'charming', 'pleasant',
        'brilliant', 'superb', 'magnificent', 'splendid', 'marvelous', 'terrific', 'fabulous',
        'positive', 'optimistic', 'hopeful', 'enthusiastic', 'passionate', 'proud', 'satisfied',
        'comfortable', 'relaxed', 'peaceful', 'content', 'fulfilled', 'accomplished',
        'חחח', 'חהחה', 'ההה', 'לול', 'אהבה', 'שמח', 'שמחה', 'כיף', 'מצחיק', 'נהדר', 'מעולה',
        'יופי', 'סבבה', 'אחלה', 'מדהים', 'וואו', 'יאי', 'תודה', 'מקסים', 'חמוד', 'כיף',
        'אהבתי', 'נפלא', 'משגע', 'מושלם', 'נהנה', 'נהנית', 'טוב', 'יפה', 'מצוין'
    }

    ANGRY_WORDS = {
        'angry', 'mad', 'furious', 'rage', 'pissed', 'annoyed', 'irritated', 'frustrated', 'upset',
        'hate', 'damn', 'dammit', 'hell', 'wtf', 'ffs', 'bullshit', 'crap', 'stupid', 'idiot',
        'ridiculous', 'unacceptable', 'outrageous', 'disgusting', 'terrible', 'horrible', 'awful',
        'unbelievable', 'seriously', 'kidding me', 'sick of', 'fed up', 'enough', 'stop',
        'annoying', 'bothering', 'disturbing', 'infuriating', 'aggravating', 'exasperating',
        'hostile', 'bitter', 'resentful', 'offended', 'insulted', 'disrespected',
        'unfair', 'unjust', 'wrong', 'rude', 'mean', 'nasty', 'cruel', 'harsh',
        'pathetic', 'worthless', 'useless', 'incompetent', 'foolish', 'moronic',
        'כועס', 'כעס', 'עצבני', 'מעצבן', 'נמאס', 'די', 'מספיק', 'זעם', 'שנאה', 'מתוסכל',
        'עזוב', 'חרא', 'זבל', 'מטומטם', 'דביל', 'אידיוט', 'בולשיט', 'מניאק', 'משוגע',
        'מרגיז', 'מתסכל', 'נורא', 'איום', 'גרוע', 'מחריד', 'מזעזע'
    }

    SAD_WORDS = {
        'sad', 'unhappy', 'depressed', 'down', 'blue', 'cry', 'crying', 'tears', 'sorry', 'apologize',
        'unfortunately', 'sadly', 'regret', 'miss', 'missed', 'lonely', 'alone', 'hurt', 'pain',
        'disappointed', 'disappointment', 'failed', 'failure', 'lost', 'loss', 'grief', 'mourn',
        'heartbroken', 'devastated', 'hopeless', 'helpless', 'tired', 'exhausted', 'drained',
        'difficult', 'hard', 'tough', 'struggle', 'struggling', 'suffered', 'suffering',
        'melancholy', 'gloomy', 'miserable', 'dejected', 'despondent', 'discouraged',
        'unfortunate', 'tragic', 'terrible', 'awful', 'horrible', 'dreadful',
        'empty', 'hollow', 'numb', 'broken', 'shattered', 'crushed', 'defeated',
        'guilty', 'ashamed', 'embarrassed', 'humiliated', 'rejected', 'abandoned',
        'worried', 'anxious', 'stressed', 'overwhelmed', 'burden', 'heavy',
        'עצוב', 'עצובה', 'בוכה', 'דמעות', 'מצטער', 'מצטערת', 'חבל', 'כאב', 'כואב', 'קשה',
        'בדידות', 'בודד', 'מאוכזב', 'נכשל', 'הפסד', 'פספוס', 'עייף', 'מותש', 'מיואש',
        'דכאון', 'דיכאון', 'שבור', 'הרוס', 'מפחד', 'חושש', 'דואג', 'מודאג', 'לחוץ'
    }

    try:
        text_lower = text_input.lower()

        happy_count = sum(1 for word in HAPPY_WORDS if word in text_lower)
        angry_count = sum(1 for word in ANGRY_WORDS if word in text_lower)
        sad_count = sum(1 for word in SAD_WORDS if word in text_lower)

        laughter_patterns = ['haha', 'hehe', 'hihi', 'hahaha', 'lol', 'lmao', 'rofl', 'חחח', 'ההה']
        has_laughter = any(pattern in text_lower for pattern in laughter_patterns)

        if has_laughter:
            dist = {"happy": 0.7, "angry": 0.1, "sad": 0.1, "neutral": 0.1}
            reason = "Laughter detected"

            if not is_hebrew and config.text_emotion_pipeline:
                try:
                    if len(text_input) > 200:
                        text_input_truncated = text_input[:200]
                    else:
                        text_input_truncated = text_input

                    results = config.text_emotion_pipeline(text_input_truncated)[0]

                    for score in results:
                        label = score['label']
                        confidence = score['score'] * 0.3

                        if label == 'anger' and confidence > 0.15:
                            dist['angry'] += confidence * 0.2
                            dist['happy'] -= confidence * 0.1
                        elif label == 'sadness' and confidence > 0.15:
                            dist['sad'] += confidence * 0.2
                            dist['happy'] -= confidence * 0.1

                except Exception:
                    pass

        elif happy_count > 0 and happy_count > angry_count and happy_count > sad_count:
            dist = {"happy": 0.4, "angry": 0.1, "sad": 0.1, "neutral": 0.4}
            reason = f"Happy words detected ({happy_count})"
        elif angry_count > 0 and angry_count > happy_count and angry_count > sad_count:
            dist = {"happy": 0.1, "angry": 0.4, "sad": 0.1, "neutral": 0.4}
            reason = f"Angry words detected ({angry_count})"
        elif sad_count > 0 and sad_count > happy_count and sad_count > angry_count:
            dist = {"happy": 0.1, "angry": 0.1, "sad": 0.4, "neutral": 0.4}
            reason = f"Sad words detected ({sad_count})"
        else:
            dist = {"happy": 0.15, "angry": 0.15, "sad": 0.15, "neutral": 0.55}
            reason = None

        if is_hebrew and hasattr(config, 'hebrew_sentiment_pipeline') and config.hebrew_sentiment_pipeline:
            try:
                hebrew_results = config.hebrew_sentiment_pipeline(text_input)[0]
                label = hebrew_results['label']
                score = hebrew_results['score']

                if label == 'positive' and score > 0.7:
                    dist['happy'] += 0.3
                    dist['neutral'] -= 0.2
                    dist['sad'] *= 0.5
                    dist['angry'] *= 0.5
                elif label == 'positive':
                    dist['happy'] += 0.15
                    dist['neutral'] -= 0.1
                elif label == 'negative' and score > 0.7:
                    if angry_count > sad_count:
                        dist['angry'] += 0.3
                        dist['sad'] += 0.1
                    else:
                        dist['sad'] += 0.3
                        dist['angry'] += 0.1
                    dist['neutral'] -= 0.3
                    dist['happy'] *= 0.5
                elif label == 'negative':
                    dist['sad'] += 0.15
                    dist['angry'] += 0.1
                    dist['neutral'] -= 0.15

                dominant_emotion = max(dist.items(), key=lambda x: x[1])[0]
                reason = f"Hebrew text: {label} ({score:.2f}), suggests '{dominant_emotion}'"

            except Exception:
                pass

        elif not has_laughter and config.text_emotion_pipeline:
            if len(text_input) > 200:
                text_input = text_input[:200]

            results = config.text_emotion_pipeline(text_input)[0]

            for score in results:
                label = score['label']
                confidence = score['score']

                if confidence < 0.2:
                    continue

                if confidence > max_emotion_score:
                    max_emotion_score = confidence
                    dominant_emotion = label

                emotion_value = confidence * 0.5

                if label == 'joy':
                    dist['happy'] += emotion_value
                    dist['neutral'] -= emotion_value * 0.8

                elif label == 'optimism':
                    dist['happy'] += emotion_value
                    dist['neutral'] -= emotion_value * 0.8

                elif label == 'love':
                    dist['happy'] += emotion_value * 0.9
                    dist['neutral'] -= emotion_value * 0.7

                elif label == 'anger':
                    if confidence > 0.45:
                        dist['angry'] += emotion_value * 0.5
                        dist['neutral'] -= emotion_value * 0.3

                elif label == 'sadness':
                    if confidence > 0.25:
                        dist['sad'] += emotion_value
                        dist['neutral'] -= emotion_value * 0.7

                elif label == 'fear':
                    dist['sad'] += emotion_value * 0.7
                    dist['angry'] += emotion_value * 0.3
                    dist['neutral'] -= emotion_value * 0.8

                elif label == 'surprise':
                    dist['happy'] += emotion_value * 0.4
                    dist['neutral'] += emotion_value * 0.6

                elif label == 'disgust':
                    dist['angry'] += emotion_value * 0.6
                    dist['sad'] += emotion_value * 0.4
                    dist['neutral'] -= emotion_value * 0.8

            if dominant_emotion and max_emotion_score > 0:
                reason = f"Text suggests '{dominant_emotion}' (conf: {max_emotion_score:.2f})"

        for key in dist:
            dist[key] = max(0.05, dist[key])

        total = sum(dist.values())
        final_dist = {k: v / total for k, v in dist.items()}

        for emo in ['happy', 'sad', 'angry']:
            if final_dist[emo] > 0.5:
                excess = final_dist[emo] - 0.5
                final_dist[emo] = 0.5
                final_dist['neutral'] += excess * 0.5

        total = sum(final_dist.values())
        final_dist = {k: v / total for k, v in final_dist.items()}

        final_emotion = max(final_dist.items(), key=lambda x: x[1])[0]
        final_confidence = final_dist[final_emotion]

        if reason is None or (
                not has_laughter and not any(count > 0 for count in [happy_count, angry_count, sad_count])):
            if final_emotion != 'neutral':
                reason = f"Text suggests '{final_emotion}' (conf: {final_confidence:.2f})"
            else:
                reason = "Text unclear"
        elif not has_laughter:
            if "Happy words" in reason and final_emotion != 'happy':
                reason = f"Text suggests '{final_emotion}' (conf: {final_confidence:.2f})"
            elif "Angry words" in reason and final_emotion != 'angry':
                reason = f"Text suggests '{final_emotion}' (conf: {final_confidence:.2f})"
            elif "Sad words" in reason and final_emotion != 'sad':
                reason = f"Text suggests '{final_emotion}' (conf: {final_confidence:.2f})"

        return final_dist, reason

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"neutral": 0.6, "happy": 0.13, "sad": 0.13, "angry": 0.14}, f"Text Error: {e}"


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

    neutral_dominance_count = 0
    if cnn_dist.get('neutral', 0) > 0.4:
        neutral_dominance_count += 1
    if logic_dist.get('neutral', 0) > 0.3:
        neutral_dominance_count += 1
    if text_dist.get('neutral', 0) > 0.4:
        neutral_dominance_count += 1

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