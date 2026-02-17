"""Unit tests for config module."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class TestConfigConstants:
    """Tests for config constants and default values."""

    def test_target_emotions_defined(self):
        assert config.TARGET_EMOTIONS == ["neutral", "angry", "happy", "sad"]

    def test_emotion_weights_sum_to_one(self):
        total = sum(config.INITIAL_EMOTION_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01

    def test_emotion_colors_defined(self):
        for emo in config.TARGET_EMOTIONS:
            assert emo in config.EMO_COL

    def test_audio_thresholds_ordered(self):
        assert config.EXTERNAL_AUDIO_VERY_LOW_RMS < config.EXTERNAL_AUDIO_HIGH_RMS
        assert config.EXTERNAL_AUDIO_HIGH_RMS < config.EXTERNAL_AUDIO_VERY_HIGH_RMS
        assert config.INTERNAL_AUDIO_VERY_LOW_RMS < config.INTERNAL_AUDIO_HIGH_RMS
        assert config.INTERNAL_AUDIO_HIGH_RMS < config.INTERNAL_AUDIO_VERY_HIGH_RMS

    def test_sample_rates_positive(self):
        assert config.PEAK_SR > 0
        assert config.VAD_SR > 0

    def test_max_gui_speakers_positive(self):
        assert config.MAX_GUI_SPK > 0

    def test_chunk_sec_positive(self):
        assert config.CHUNK_SEC > 0

    def test_vad_threshold_in_range(self):
        assert 0 < config.SILERO_VAD_THRESHOLD < 1

    def test_distress_window_positive(self):
        assert config.DISTRESS_WINDOW_SECONDS > 0

    def test_emotion_sensitivity_positive(self):
        assert config.EMOTION_SENSITIVITY > 0

    def test_spam_confidence_threshold_in_range(self):
        assert 0 < config.SPAM_CONFIDENCE_THRESHOLD < 1

    def test_spam_mismatch_pairs_are_tuples(self):
        for pair in config.SPAM_MISMATCH_PAIRS:
            assert isinstance(pair, tuple)
            assert len(pair) == 2

    def test_initial_caches_correct_size(self):
        assert len(config._last_significant_emotion_cache_spk) == config.MAX_GUI_SPK
        assert len(config._current_1s_emotion_cache_spk) == config.MAX_GUI_SPK

    def test_distress_history_initialized(self):
        assert len(config.distress_detection_history) == config.MAX_GUI_SPK
        assert len(config.current_distress_status) == config.MAX_GUI_SPK

    def test_thread_pool_executors_exist(self):
        assert config.stt_executor is not None
        assert config.emotion_executor is not None
