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
    "ok_distance_thresh":      0.055,
    "ok_confirm_frames":        3,
    "ok_cooldown_frames":        6,
    "gesture_scroll_confirm_frames": 2,
    "scroll_smooth":            0.35,
    "scroll_reset_deadzone":    0.004,
    "scroll_step":              90,
    "scroll_trigger":           0.02,
    "scroll_trigger_x":         0.015,
    "scroll_trigger_y":         0.02,
    "scroll_hold_start":        0.03,
    "scroll_hold_interval_frames": 6,
    "scroll_axis_bias_x":       1.35,
    "scroll_use_shift_hscroll": True,
    "scroll_invert_x":          False,
    "scroll_invert_y":          False,
    "gesture_scroll_smooth": 0.4,
    "gesture_scroll_min_scroll": 1,

    # Voice
    "mic_energy_threshold": 300,
    "voice_speaking_secs":  2,
    "voice_listen_always_on": True,
    "voice_idle_stop_secs": 300,
    "voice_vad_block_ms": 40,
    "voice_vad_speech_mult": 2.2,
    "voice_vad_silence_mult": 1.2,
    "voice_vad_hangover_frames": 8,
    "voice_vad_min_ms": 400,
    "voice_vad_max_secs": 6,

    # AI decision
    "ai_action_cooldown_frames": 15,
    "start_paused": True,

    # Web search (Serper)
    "serper_api_key": "",
    "serper_auto_search": False,
    "serper_max_results": 5,
    "serper_cache_ttl_secs": 900,
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

    def set_many(self, updates: dict):
        for k, v in updates.items():
            self._data[k] = v

    def all(self) -> dict:
        return dict(self._data)

    def save(self, path: str = "config/settings.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
