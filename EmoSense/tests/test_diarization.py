"""Unit tests for diarization module."""

import numpy as np
import os
import sys
import time
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class TestOnlineDia:
    """Tests for the OnlineDia speaker diarization class."""

    def test_first_speaker_gets_id_0(self):
        from diarization import OnlineDia
        dia = OnlineDia()
        emb = np.random.randn(256).astype(np.float32)
        speaker_id = dia.add(emb)
        assert speaker_id == 0

    def test_same_speaker_returns_same_id(self):
        from diarization import OnlineDia
        dia = OnlineDia()
        emb = np.random.randn(256).astype(np.float32)

        id1 = dia.add(emb)
        id2 = dia.add(emb + np.random.randn(256) * 0.01)  # small noise
        assert id1 == id2

    def test_different_speakers_get_different_ids(self):
        from diarization import OnlineDia
        dia = OnlineDia()

        emb1 = np.random.randn(256).astype(np.float32)
        emb2 = -emb1  # opposite direction = maximum cosine distance

        id1 = dia.add(emb1)
        id2 = dia.add(emb2)
        assert id1 != id2

    def test_prune_removes_old_speakers(self):
        from diarization import OnlineDia
        dia = OnlineDia(max_inactive_time=0.01)
        emb = np.random.randn(256).astype(np.float32)
        dia.add(emb)

        time.sleep(0.02)
        dia.prune_speakers()
        assert len(dia.centroids) == 0

    def test_prune_keeps_recent_speakers(self):
        from diarization import OnlineDia
        dia = OnlineDia(max_inactive_time=100)
        emb = np.random.randn(256).astype(np.float32)
        dia.add(emb)

        dia.prune_speakers()
        assert len(dia.centroids) == 1

    def test_centroid_update_with_momentum(self):
        from diarization import OnlineDia
        dia = OnlineDia(momentum=0.5)

        emb1 = np.ones(256, dtype=np.float32)
        dia.add(emb1)
        original_centroid = dia.centroids[0].copy()

        emb2 = emb1 + np.ones(256, dtype=np.float32) * 0.01
        dia.add(emb2)

        assert not np.array_equal(dia.centroids[0], original_centroid)


class TestGuiSpeakerMapper:
    """Tests for the GuiSpeakerMapper class."""

    def test_first_speaker_maps_to_slot(self):
        from diarization import GuiSpeakerMapper
        mapper = GuiSpeakerMapper(num_gui_slots=2)
        emb = np.random.randn(256).astype(np.float32)
        gui_id = mapper.get_gui_sid(0, emb)
        assert 0 <= gui_id < 2

    def test_same_internal_id_returns_same_gui_id(self):
        from diarization import GuiSpeakerMapper
        mapper = GuiSpeakerMapper(num_gui_slots=2)
        emb = np.random.randn(256).astype(np.float32)

        gui_id1 = mapper.get_gui_sid(0, emb)
        gui_id2 = mapper.get_gui_sid(0, emb)
        assert gui_id1 == gui_id2

    def test_different_speakers_get_different_slots(self):
        from diarization import GuiSpeakerMapper
        mapper = GuiSpeakerMapper(num_gui_slots=2)

        gui_id1 = mapper.get_gui_sid(0, np.random.randn(256))
        gui_id2 = mapper.get_gui_sid(1, np.random.randn(256))
        assert gui_id1 != gui_id2

    def test_slot_recycling_when_full(self):
        from diarization import GuiSpeakerMapper
        mapper = GuiSpeakerMapper(num_gui_slots=2)

        mapper.get_gui_sid(0, np.random.randn(256))
        mapper.get_gui_sid(1, np.random.randn(256))

        time.sleep(0.1)
        gui_id3 = mapper.get_gui_sid(2, np.random.randn(256))
        assert 0 <= gui_id3 < 2

    def test_get_speaker_info_returns_none(self):
        from diarization import GuiSpeakerMapper
        mapper = GuiSpeakerMapper()
        assert mapper.get_speaker_info(0) is None


class TestAcousticFeatureExtractor:
    """Tests for the AcousticFeatureExtractor class."""

    def test_extract_features_valid_audio(self):
        try:
            from diarization import AcousticFeatureExtractor
        except ImportError:
            pytest.skip("librosa not available")

        extractor = AcousticFeatureExtractor()
        audio = np.random.randn(16000).astype(np.float32)
        features = extractor.extract_features(audio)

        assert len(features) == 2
        assert features[0] >= 0  # RMS is always positive

    def test_extract_features_silent_audio(self):
        try:
            from diarization import AcousticFeatureExtractor
        except ImportError:
            pytest.skip("librosa not available")

        extractor = AcousticFeatureExtractor()
        audio = np.zeros(16000, dtype=np.float32)
        features = extractor.extract_features(audio)
        assert features[0] == 0.0
