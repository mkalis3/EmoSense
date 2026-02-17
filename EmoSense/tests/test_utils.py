"""Unit tests for utils module."""

import json
import os
import tempfile
import pytest

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestContainsPredominantlyHebrew:
    """Tests for the Hebrew text detection function."""

    def test_pure_hebrew_text(self):
        from utils import contains_predominantly_hebrew
        assert contains_predominantly_hebrew("שלום עולם") is True

    def test_pure_english_text(self):
        from utils import contains_predominantly_hebrew
        assert contains_predominantly_hebrew("Hello world") is False

    def test_empty_string(self):
        from utils import contains_predominantly_hebrew
        assert contains_predominantly_hebrew("") is False

    def test_none_input(self):
        from utils import contains_predominantly_hebrew
        assert contains_predominantly_hebrew(None) is False

    def test_mixed_text_hebrew_dominant(self):
        from utils import contains_predominantly_hebrew
        assert contains_predominantly_hebrew("שלום hello שלום עולם") is True

    def test_mixed_text_english_dominant(self):
        from utils import contains_predominantly_hebrew
        assert contains_predominantly_hebrew("hello world שלום") is False

    def test_numbers_only(self):
        from utils import contains_predominantly_hebrew
        assert contains_predominantly_hebrew("12345") is False

    def test_single_hebrew_char(self):
        from utils import contains_predominantly_hebrew
        assert contains_predominantly_hebrew("א") is True

    def test_whitespace_only(self):
        from utils import contains_predominantly_hebrew
        assert contains_predominantly_hebrew("   ") is False


class TestSaveSettings:
    """Tests for the save_settings function."""

    def test_save_valid_settings(self, tmp_path):
        import config
        original_path = config.SETTINGS_FILE
        config.SETTINGS_FILE = str(tmp_path / "test_settings.json")

        from utils import save_settings
        settings = {"weights": {"cnn": 0.35, "logic": 0.35, "text": 0.30}}
        save_settings(settings)

        with open(config.SETTINGS_FILE, 'r') as f:
            loaded = json.load(f)
        assert loaded == settings

        config.SETTINGS_FILE = original_path

    def test_save_empty_settings(self, tmp_path):
        import config
        original_path = config.SETTINGS_FILE
        config.SETTINGS_FILE = str(tmp_path / "test_settings.json")

        from utils import save_settings
        save_settings({})

        with open(config.SETTINGS_FILE, 'r') as f:
            loaded = json.load(f)
        assert loaded == {}

        config.SETTINGS_FILE = original_path


class TestLoadSettings:
    """Tests for the load_settings function."""

    def test_load_default_when_no_file(self, tmp_path):
        import config
        original_path = config.SETTINGS_FILE
        config.SETTINGS_FILE = str(tmp_path / "nonexistent.json")

        from utils import load_settings
        settings = load_settings()

        assert 'weights' in settings
        assert 'audio_source' in settings
        assert settings['weights'] == config.INITIAL_EMOTION_WEIGHTS

        config.SETTINGS_FILE = original_path

    def test_load_existing_settings(self, tmp_path):
        import config
        original_path = config.SETTINGS_FILE
        settings_file = tmp_path / "test_settings.json"

        test_data = {
            "weights": {"cnn": 0.5, "logic": 0.3, "text": 0.2},
            "audio_source": "External Microphone",
            "internal_device_name": "test_device"
        }
        with open(settings_file, 'w') as f:
            json.dump(test_data, f)

        config.SETTINGS_FILE = str(settings_file)
        from utils import load_settings
        settings = load_settings()

        assert settings['weights'] == test_data['weights']
        assert settings['audio_source'] == test_data['audio_source']

        config.SETTINGS_FILE = original_path

    def test_load_corrupt_json(self, tmp_path):
        import config
        original_path = config.SETTINGS_FILE
        settings_file = tmp_path / "corrupt.json"
        settings_file.write_text("{invalid json")

        config.SETTINGS_FILE = str(settings_file)
        from utils import load_settings
        settings = load_settings()

        assert 'weights' in settings
        assert settings['weights'] == config.INITIAL_EMOTION_WEIGHTS

        config.SETTINGS_FILE = original_path

    def test_removes_intense_mode(self, tmp_path):
        import config
        original_path = config.SETTINGS_FILE
        settings_file = tmp_path / "settings_with_intense.json"

        test_data = {
            "weights": {"cnn": 0.35, "logic": 0.35, "text": 0.30},
            "audio_source": "External Microphone",
            "internal_device_name": "",
            "intense_mode": True
        }
        with open(settings_file, 'w') as f:
            json.dump(test_data, f)

        config.SETTINGS_FILE = str(settings_file)
        from utils import load_settings
        settings = load_settings()
        assert 'intense_mode' not in settings

        config.SETTINGS_FILE = original_path
