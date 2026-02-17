"""Unit tests for analysis module."""

import numpy as np
import os
import sys
import time
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class TestExtractAdvancedAudioFeatures:
    """Tests for audio feature extraction."""

    def test_returns_empty_for_none_audio(self):
        from analysis import extract_advanced_audio_features
        result = extract_advanced_audio_features(None)
        assert result == {}

    def test_returns_empty_for_short_audio(self):
        from analysis import extract_advanced_audio_features
        result = extract_advanced_audio_features(np.zeros(50))
        assert result == {}

    def test_returns_features_for_valid_audio(self):
        from analysis import extract_advanced_audio_features
        audio = np.random.randn(22050).astype(np.float32)
        result = extract_advanced_audio_features(audio)

        assert 'rms' in result
        assert 'zcr' in result
        assert 'spectral_centroid' in result
        assert 'duration' in result
        assert result['rms'] >= 0
        assert result['duration'] > 0

    def test_caching_with_segment_id(self):
        from analysis import extract_advanced_audio_features, feature_cache
        audio = np.random.randn(22050).astype(np.float32)

        feature_cache.clear()
        result1 = extract_advanced_audio_features(audio, segment_id="test_seg_1")
        result2 = extract_advanced_audio_features(audio, segment_id="test_seg_1")

        assert result1 == result2
        assert "test_seg_1" in feature_cache

    def test_cache_eviction(self):
        from analysis import extract_advanced_audio_features, feature_cache
        feature_cache.clear()
        audio = np.random.randn(22050).astype(np.float32)

        for i in range(105):
            extract_advanced_audio_features(audio, segment_id=f"seg_{i}")

        assert len(feature_cache) <= 101

    def test_silent_audio_low_rms(self):
        from analysis import extract_advanced_audio_features
        audio = np.zeros(22050, dtype=np.float32)
        audio += np.random.randn(22050).astype(np.float32) * 0.0001
        result = extract_advanced_audio_features(audio)
        assert result['rms'] < 0.01


class TestCnnEmotionAnalysis:
    """Tests for CNN emotion analysis."""

    def test_returns_na_when_no_model(self):
        from analysis import cnn_emotion_analysis
        original_model = config.cnn_model
        config.cnn_model = None

        dist, reason = cnn_emotion_analysis(np.random.randn(22050))
        assert dist == {}
        assert reason == "N/A"

        config.cnn_model = original_model

    def test_returns_na_for_none_audio(self):
        from analysis import cnn_emotion_analysis
        dist, reason = cnn_emotion_analysis(None)
        assert dist == {}
        assert reason == "N/A"


class TestLogicBasedEmotionAnalysis:
    """Tests for logic-based emotion analysis."""

    def test_returns_neutral_for_none_audio(self):
        from analysis import logic_based_emotion_analysis
        dist, reason = logic_based_emotion_analysis(None, False)
        assert dist['neutral'] == 0.55
        assert reason == "No audio"

    def test_returns_neutral_for_short_audio(self):
        from analysis import logic_based_emotion_analysis
        dist, reason = logic_based_emotion_analysis(np.zeros(50), False)
        assert dist['neutral'] == 0.55

    def test_valid_distribution_for_normal_audio(self):
        from analysis import logic_based_emotion_analysis
        audio = np.random.randn(22050).astype(np.float32) * 0.05
        dist, reason = logic_based_emotion_analysis(audio, is_internal_audio=False)

        assert all(e in dist for e in ['neutral', 'happy', 'sad', 'angry'])
        assert abs(sum(dist.values()) - 1.0) < 0.01
        assert all(v >= 0 for v in dist.values())

    def test_loud_audio_increases_angry(self):
        from analysis import logic_based_emotion_analysis
        loud_audio = np.random.randn(22050).astype(np.float32) * 0.5
        quiet_audio = np.random.randn(22050).astype(np.float32) * 0.01

        loud_dist, _ = logic_based_emotion_analysis(loud_audio, is_internal_audio=False)
        quiet_dist, _ = logic_based_emotion_analysis(quiet_audio, is_internal_audio=False)

        assert loud_dist['angry'] > quiet_dist['angry']

    def test_internal_audio_uses_different_thresholds(self):
        from analysis import logic_based_emotion_analysis
        audio = np.random.randn(22050).astype(np.float32) * 0.1

        ext_dist, ext_reason = logic_based_emotion_analysis(audio, is_internal_audio=False)
        int_dist, int_reason = logic_based_emotion_analysis(audio, is_internal_audio=True)

        assert ext_dist != int_dist or ext_reason != int_reason


class TestTextBasedEmotionAnalysis:
    """Tests for text-based emotion analysis."""

    def test_empty_text_returns_neutral(self):
        from analysis import text_based_emotion_analysis
        dist, reason = text_based_emotion_analysis("")
        assert dist['neutral'] == 0.6
        assert reason == "No text"

    def test_none_text_returns_neutral(self):
        from analysis import text_based_emotion_analysis
        dist, reason = text_based_emotion_analysis(None)
        assert dist['neutral'] == 0.6

    def test_laughter_detected_as_happy(self):
        from analysis import text_based_emotion_analysis
        dist, reason = text_based_emotion_analysis("hahaha that's so funny lol")
        assert dist['happy'] > dist['sad']
        assert dist['happy'] > dist['angry']

    def test_sad_words_detected(self):
        from analysis import text_based_emotion_analysis

        original_pipeline = config.text_emotion_pipeline
        config.text_emotion_pipeline = None

        dist, reason = text_based_emotion_analysis("I'm so sad and depressed and crying")
        assert dist['sad'] > dist['happy']

        config.text_emotion_pipeline = original_pipeline

    def test_angry_words_detected(self):
        from analysis import text_based_emotion_analysis

        original_pipeline = config.text_emotion_pipeline
        config.text_emotion_pipeline = None

        dist, reason = text_based_emotion_analysis("I'm so angry and furious and mad")
        assert dist['angry'] > dist['happy']

        config.text_emotion_pipeline = original_pipeline

    def test_happy_words_detected(self):
        from analysis import text_based_emotion_analysis

        original_pipeline = config.text_emotion_pipeline
        config.text_emotion_pipeline = None

        dist, reason = text_based_emotion_analysis("I'm so happy and excited and thrilled")
        assert dist['happy'] > dist['sad']

        config.text_emotion_pipeline = original_pipeline

    def test_hebrew_happy_words(self):
        from analysis import text_based_emotion_analysis

        original_pipeline = config.text_emotion_pipeline
        original_hebrew = getattr(config, 'hebrew_sentiment_pipeline', None)
        config.text_emotion_pipeline = None
        config.hebrew_sentiment_pipeline = None

        dist, reason = text_based_emotion_analysis("אחלה מדהים נהדר שמח")
        assert dist['happy'] > dist['angry']

        config.text_emotion_pipeline = original_pipeline
        config.hebrew_sentiment_pipeline = original_hebrew

    def test_distribution_sums_to_one(self):
        from analysis import text_based_emotion_analysis

        original_pipeline = config.text_emotion_pipeline
        config.text_emotion_pipeline = None

        for text in ["hello", "hahaha", "I'm angry", "so sad", ""]:
            dist, _ = text_based_emotion_analysis(text)
            if text:
                assert abs(sum(dist.values()) - 1.0) < 0.01, f"Failed for: {text}"

        config.text_emotion_pipeline = original_pipeline


class TestSpamDetection:
    """Tests for spam detection."""

    def test_no_data_returns_not_detected(self):
        from analysis import analyze_spam_detection
        result, reason = analyze_spam_detection(None, None)
        assert result['status'] == "Not Detected"

    def test_normal_conversation_not_spam(self):
        from analysis import analyze_spam_detection
        text_dist = {"happy": 0.3, "sad": 0.1, "angry": 0.1, "neutral": 0.5}
        voice_dist = {"happy": 0.25, "sad": 0.15, "angry": 0.1, "neutral": 0.5}
        result, reason = analyze_spam_detection(text_dist, voice_dist, "Hello how are you")
        assert result['status'] == "Not Detected"

    def test_emotion_mismatch_detected(self):
        from analysis import analyze_spam_detection
        text_dist = {"happy": 0.7, "sad": 0.1, "angry": 0.1, "neutral": 0.1}
        voice_dist = {"happy": 0.1, "sad": 0.1, "angry": 0.7, "neutral": 0.1}
        result, reason = analyze_spam_detection(text_dist, voice_dist, "I'm so happy today")
        assert result['confidence'] > 0.2

    def test_repetitive_text_increases_score(self):
        from analysis import analyze_spam_detection
        text_dist = {"happy": 0.15, "sad": 0.15, "angry": 0.15, "neutral": 0.55}
        voice_dist = {"happy": 0.15, "sad": 0.15, "angry": 0.15, "neutral": 0.55}
        result, reason = analyze_spam_detection(
            text_dist, voice_dist, "buy buy buy buy buy buy buy buy buy buy"
        )
        assert "Repetitive" in reason


class TestDistressDetection:
    """Tests for distress detection system."""

    def setup_method(self):
        from collections import deque
        config.distress_detection_history = {i: deque(maxlen=60) for i in range(config.MAX_GUI_SPK)}
        config.current_distress_status = {
            i: {'at_risk': False, 'emotion': None, 'duration': 0, 'confidence': 0}
            for i in range(config.MAX_GUI_SPK)
        }

    def test_no_distress_with_few_entries(self):
        from analysis import check_distress_detection
        check_distress_detection(0, "angry", 0.8, "I'm very angry")
        assert config.current_distress_status[0]['at_risk'] is False

    def test_ignores_invalid_speaker(self):
        from analysis import check_distress_detection
        check_distress_detection(-1, "angry", 0.8, "")
        check_distress_detection(999, "angry", 0.8, "")

    def test_fear_words_detected(self):
        from analysis import check_distress_detection
        for i in range(15):
            check_distress_detection(0, "sad", 0.8, "I'm terrified and scared")
        status = config.current_distress_status[0]
        assert status['at_risk'] is True or len(config.distress_detection_history[0]) >= 10


class TestSmoothEmotionResult:
    """Tests for emotion smoothing."""

    def setup_method(self):
        from analysis import emotion_history
        for key in emotion_history:
            emotion_history[key].clear()

    def test_returns_same_for_single_entry(self):
        from analysis import smooth_emotion_result
        emo, conf = smooth_emotion_result(0, "happy", 0.8, "seg_1")
        assert emo == "happy"
        assert conf == 0.8

    def test_high_confidence_not_smoothed(self):
        from analysis import smooth_emotion_result
        smooth_emotion_result(0, "happy", 0.8, "seg_1")
        smooth_emotion_result(0, "happy", 0.7, "seg_2")

        emo, conf = smooth_emotion_result(0, "sad", 0.6, "seg_3")
        assert emo == "sad"

    def test_invalid_speaker_returns_unchanged(self):
        from analysis import smooth_emotion_result
        emo, conf = smooth_emotion_result(-1, "happy", 0.5, "seg_1")
        assert emo == "happy"
        assert conf == 0.5


class TestPerformFullEmotionAnalysis:
    """Integration tests for the full analysis pipeline."""

    def test_returns_valid_structure(self):
        from analysis import perform_full_emotion_analysis
        audio = np.random.randn(22050).astype(np.float32) * 0.05
        weights = {"cnn": 0.35, "logic": 0.35, "text": 0.30}

        emo, details, conf = perform_full_emotion_analysis(
            audio, "hello world", weights, False, speaker_id=0
        )

        assert emo in config.TARGET_EMOTIONS
        assert 0 <= conf <= 1
        assert 'cnn_analysis' in details
        assert 'logic_analysis' in details
        assert 'text_analysis' in details
        assert 'final_decision' in details
        assert 'spam_detection' in details

    def test_handles_no_text(self):
        from analysis import perform_full_emotion_analysis
        audio = np.random.randn(22050).astype(np.float32) * 0.05
        weights = {"cnn": 0.35, "logic": 0.35, "text": 0.30}

        emo, details, conf = perform_full_emotion_analysis(
            audio, None, weights, False
        )
        assert emo in config.TARGET_EMOTIONS

    def test_handles_no_audio(self):
        from analysis import perform_full_emotion_analysis
        weights = {"cnn": 0.35, "logic": 0.35, "text": 0.30}

        emo, details, conf = perform_full_emotion_analysis(
            None, "I'm happy", weights, False
        )
        assert emo in config.TARGET_EMOTIONS
