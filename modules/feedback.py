"""
modules/feedback.py  —  Smart Feedback System
"""

import threading
import time

try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False

try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    _AUDIO_AVAILABLE = True
except Exception:
    _AUDIO_AVAILABLE = False

import numpy as np


class FeedbackSystem:
    def __init__(self):
        self._tts_engine = None
        self._tts_lock   = threading.Lock()
        self._flash_state = {}

        if _TTS_AVAILABLE:
            try:
                self._tts_engine = pyttsx3.init()
                self._tts_engine.setProperty("rate", 175)
                self._tts_engine.setProperty("volume", 0.85)
            except Exception:
                self._tts_engine = None

    def speak(self, text: str):
        """Non-blocking TTS."""
        if self._tts_engine is None:
            print(f"[Feedback] {text}")
            return
        def _run():
            with self._tts_lock:
                self._tts_engine.say(text)
                self._tts_engine.runAndWait()
        threading.Thread(target=_run, daemon=True).start()

    def beep(self, freq: int = 440, duration_ms: int = 100):
        """Generate a pure tone beep."""
        if not _AUDIO_AVAILABLE:
            return
        def _play():
            sr     = 44100
            frames = int(sr * duration_ms / 1000)
            t      = np.linspace(0, duration_ms / 1000, frames, False)
            wave   = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)
            init = pygame.mixer.get_init()
            channels = init[2] if init else 1
            if channels > 1:
                wave = np.repeat(wave[:, None], channels, axis=1)
            sound  = pygame.sndarray.make_sound(wave)
            sound.play()
            time.sleep(duration_ms / 1000 + 0.05)
        threading.Thread(target=_play, daemon=True).start()

    def visual_flash(self, event_type: str):
        """Record flash event (drawn in overlay by caller if needed)."""
        self._flash_state[event_type] = time.time()

    def is_flashing(self, event_type: str, duration: float = 0.3) -> bool:
        t = self._flash_state.get(event_type)
        return t is not None and (time.time() - t) < duration
