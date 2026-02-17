"""Unit tests for report_generator module."""

import os
import sys
import time
import pytest
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class TestGenerateSummaryReport:
    """Tests for report generation."""

    def test_returns_none_when_no_data(self):
        original_segments = config.segments_data
        original_map = config.segment_id_map

        config.segments_data = deque()
        config.segment_id_map = {}

        from report_generator import generate_summary_report
        result = generate_summary_report()
        assert result is None

        config.segments_data = original_segments
        config.segment_id_map = original_map

    def test_generates_report_file(self, tmp_path):
        import numpy as np
        original_segments = config.segments_data
        original_map = config.segment_id_map
        original_cwd = os.getcwd()

        os.chdir(tmp_path)

        config.segments_data = deque([
            {"id": 0, "audio": np.zeros(22050), "words": ["hello"], "sid": 0, "emotion_state": "happy"}
        ])
        config.segment_id_map = {
            0: {
                "id": 0, "audio": np.zeros(22050), "words": ["hello"],
                "sid": 0, "emotion_state": "happy", "emotion_confidence": 0.8
            }
        }

        from report_generator import generate_summary_report
        result = generate_summary_report()

        assert result is not None
        assert os.path.exists(result)

        with open(result, 'r') as f:
            content = f.read()
        assert "EMOSENSE" in content
        assert "Speaker 0" in content

        os.chdir(original_cwd)
        config.segments_data = original_segments
        config.segment_id_map = original_map

    def test_report_includes_distress_info(self, tmp_path):
        import numpy as np
        original_segments = config.segments_data
        original_map = config.segment_id_map
        original_distress = config.current_distress_status
        original_cwd = os.getcwd()

        os.chdir(tmp_path)

        config.segments_data = deque([
            {"id": 0, "audio": np.zeros(22050), "words": ["test"], "sid": 0, "emotion_state": "angry"}
        ])
        config.segment_id_map = {
            0: {"id": 0, "audio": np.zeros(22050), "words": ["test"],
                "sid": 0, "emotion_state": "angry", "emotion_confidence": 0.9}
        }
        config.current_distress_status = {
            0: {'at_risk': True, 'emotion': 'angry', 'duration': 120, 'confidence': 0.85},
            1: {'at_risk': False, 'emotion': None, 'duration': 0, 'confidence': 0}
        }

        from report_generator import generate_summary_report
        result = generate_summary_report()

        with open(result, 'r') as f:
            content = f.read()
        assert "AT RISK" in content

        os.chdir(original_cwd)
        config.segments_data = original_segments
        config.segment_id_map = original_map
        config.current_distress_status = original_distress
