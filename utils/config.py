"""
utils/config.py  —  Eyecon Configuration Manager
"""

import json
import os


_DEFAULTS = {
    # Camera
    "camera_index":   0,
    "cam_width":      640,
    "cam_height":     480,
    "cam_fps":        30,

    # Eye tracking
    "auto_calibrate":       True,
    "blink_click":          True,
    "dwell_click":          True,
    "gaze_smooth_frames":   6,
    "eye_moves_cursor":     True,
    "eye_inactive_secs":    10,

    # Gesture
    "gesture_confirm_frames":  8,
    "gesture_cooldown_frames": 20,
    "gesture_inactive_secs":   15,
    "gesture_smooth_frames":   4,
    "gesture_smooth_min_count": 3,
    "fist_confirm_frames":      6,
    "fist_min_present_frames": 3,
    "fist_edge_margin": 0.04,
    "pinch_confirm_frames":     2,
    "pinch_cooldown_frames":    6,
    "gesture_scroll_confirm_frames": 2,
    "gesture_scroll_smooth": 0.4,
    "gesture_scroll_min_scroll": 1,

    # Voice
    "mic_energy_threshold": 300,
    "voice_speaking_secs":  2,

    # AI decision
    "ai_action_cooldown_frames": 15,
    "start_paused": True,
}


class Config:
    def __init__(self, path: str = "config/settings.json"):
        self._data = dict(_DEFAULTS)
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    overrides = json.load(f)
                self._data.update(overrides)
            except Exception as e:
                print(f"[Config] Could not load {path}: {e} — using defaults")

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value):
        self._data[key] = value

    def save(self, path: str = "config/settings.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
