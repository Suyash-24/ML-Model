"""
app.py  —  Eyecon WEDNESDAY Interface  (complete rewrite)
──────────────────────────────────────────────────────────
Replaces the old blue dashboard entirely.

Layout:
  ┌─────────┬──────────────────────────────────────┐
  │         │  Top bar (40px)                       │
  │ Sidebar ├────────────┬─────────────────────────┤
  │ (152px) │ Sphere     │  Greeting + metrics      │
  │         │ (238px)    │  Conversation panel      │
  │         │            │  Input bar               │
  └─────────┴────────────┴─────────────────────────┘

Sphere: Three.js via QWebEngineView + QWebChannel bridge
Python sends state changes to sphere via JS bridge.

PLACE AT: ML-Model/app.py  (replace existing)
"""

import os, sys, time, threading, random
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QFrame, QScrollArea, QSizePolicy,
    QLineEdit, QStackedWidget,
)
from PyQt6.QtCore  import Qt, QTimer, QThread, pyqtSignal, QObject, QUrl
from PyQt6.QtGui   import QFont, QPixmap, QImage, QColor
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebChannel       import QWebChannel

try:
    from modules.eye_tracker    import EyeTracker
    from modules.gesture_engine import GestureEngine
    from modules.wednesday_engine import WednesdayEngine
    from modules.ai_decision    import AIDecisionModule
    from modules.feedback       import FeedbackSystem
    from utils.config           import Config
    BACKEND = True
except ImportError as e:
    print(f"[app] Backend unavailable ({e}) — demo mode")
    BACKEND = False


# ═══════════════════════════════════════════════════════════════════════════════
#  SPHERE BRIDGE  — Python ↔ JS
# ═══════════════════════════════════════════════════════════════════════════════
class SphereBridge(QObject):
    stateChanged = pyqtSignal(str)   # JS connects to this signal

    def set_state(self, state: str):
        self.stateChanged.emit(state)


# ═══════════════════════════════════════════════════════════════════════════════
#  CAMERA WORKER
# ═══════════════════════════════════════════════════════════════════════════════
class CameraWorker(QObject):
    frame_ready = pyqtSignal(np.ndarray, dict)

    def __init__(self, backend: bool, cfg=None, feedback=None):
        super().__init__()
        self._running  = False
        self._backend  = backend
        self._cfg      = cfg
        self._feedback = feedback
        self._cap      = None
        self._eye = self._gesture = self._wednesday = self._ai = None

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def _loop(self):
        self._cap = cv2.VideoCapture(0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS,           30)

        if self._backend:
            self._eye     = EyeTracker(self._cfg, self._feedback)
            self._gesture = GestureEngine(self._cfg, self._feedback)
            self._wednesday = WednesdayEngine(self._cfg, self._feedback)
            self._ai        = AIDecisionModule(
                self._eye, self._gesture, self._wednesday,
                self._feedback, self._cfg)
            threading.Thread(target=self._wednesday.listen_loop,
                             daemon=True).start()

        t0, frames = time.time(), 0
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.02)
                continue
            frame  = cv2.flip(frame, 1)
            frames += 1
            fps = frames / max(time.time() - t0, 1e-6)

            data = {"fps": fps, "cpu": random.randint(10, 28),
                    "latency": random.randint(14, 32),
                    "gesture": "—", "confidence": 0,
                    "eye_active": False, "hand_active": False,
                    "action": None, "source": None}

            if self._backend:
                eye_d  = self._eye.process(frame)
                gest_d = self._gesture.process(frame)
                action = self._ai.decide(eye_d, gest_d, frame)
                if action:
                    self._ai.execute(action)
                    data["action"] = action.get("action")
                    data["source"] = action.get("source")
                self._eye.draw_overlay(frame, eye_d)
                self._gesture.draw_overlay(frame, gest_d)
                self._ai.draw_status(frame, action)
                data["eye_active"]  = eye_d.get("active", False)
                data["hand_active"] = gest_d.get("active", False)
                data["gesture"]     = gest_d.get("gesture", "—") or "—"
                data["confidence"]  = int(min(gest_d.get("confidence", 0), 1) * 100)

            self.frame_ready.emit(frame, data)
            time.sleep(0.001)

        if self._cap: self._cap.release()

    def get_wednesday(self):
        return self._wednesday

    # backwards compat
    def get_friday(self):
        return self._wednesday


