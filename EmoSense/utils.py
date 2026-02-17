"""Utility functions for EmoSense settings management and text processing."""

import os
import re
import json
import config


def save_settings(settings_to_save: dict) -> None:
    """Persist application settings to JSON file.

    Args:
        settings_to_save: Dictionary of settings to serialize.
    """
    try:
        with open(config.SETTINGS_FILE, 'w') as f:
            json.dump(settings_to_save, f, indent=4)
    except Exception as e:
        print(f"Failed to save settings: {e}", flush=True)


def load_settings() -> dict:
    """Load application settings from JSON file, with defaults for missing keys.

    Returns:
        Dictionary containing weights, audio_source, and internal_device_name.
    """
    if os.path.exists(config.SETTINGS_FILE):
        try:
            with open(config.SETTINGS_FILE, 'r') as f:
                settings = json.load(f)

                if 'weights' not in settings:
                    settings['weights'] = config.INITIAL_EMOTION_WEIGHTS.copy()
                if 'audio_source' not in settings:
                    settings['audio_source'] = config.AUDIO_SOURCE_EXTERNAL
                if 'internal_device_name' not in settings:
                    settings['internal_device_name'] = ''

                if 'intense_mode' in settings:
                    del settings['intense_mode']

                return settings
        except (json.JSONDecodeError, IOError) as e:
            print(f"Failed to load settings file: {e}", flush=True)

    print("Settings file not found or corrupt. Loading default settings.", flush=True)
    return {
        'weights': config.INITIAL_EMOTION_WEIGHTS.copy(),
        'audio_source': config.AUDIO_SOURCE_EXTERNAL,
        'internal_device_name': ''
    }


def contains_predominantly_hebrew(text: str) -> bool:
    """Check if the given text is predominantly Hebrew characters.

    Args:
        text: Input text to analyze.

    Returns:
        True if more than 50% of characters are Hebrew.
    """
    if not text:
        return False
    hebrew_chars = len(re.findall(r'[\u0590-\u05FF]', text))
    return (hebrew_chars / len(text)) > 0.5 if text else False