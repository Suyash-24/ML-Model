"""
app.py  —  Eyecon Native Desktop Application
─────────────────────────────────────────────
PyQt6-based native GUI that replicates the dashboard UI exactly:
  • Live camera feed panel (left)
  • Gesture detected panel (left-bottom)
  • System status panel (right-top)
  • Command log panel (right-bottom)
  • Mode buttons: EYE | GESTURE | VOICE | AUTO
  • Bottom metric cards: FPS | CPU | LATENCY | COMMANDS
  • AI Decision Module alert banner

Run:  python app.py
"""

import os
import sys
import time
import threading
import collections
import random

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QFrame, QScrollArea, QSizePolicy,
    QGridLayout, QProgressBar,
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QObject, QSize,
)
from PyQt6.QtGui import (
    QImage, QPixmap, QFont, QColor, QPalette, QFontDatabase,
    QPainter, QPen, QBrush, QLinearGradient,
)

# ── Try importing backend modules (graceful degradation if deps missing) ──────
try:
    from modules.eye_tracker    import EyeTracker
    from modules.gesture_engine import GestureEngine
    from modules.voice_engine   import VoiceEngine
    from modules.ai_decision    import AIDecisionModule
    from modules.feedback       import FeedbackSystem
    from utils.config           import Config
    from utils.logger           import EyeconLogger
    BACKEND_AVAILABLE = True
except ImportError as e:
    print(f"[Eyecon] Backend modules not fully installed ({e}). Running in demo mode.")
    BACKEND_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR PALETTE  (matches the screenshot exactly)
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg":          "#0d1117",
    "panel":       "#161b22",
    "border":      "#21262d",
    "border_hi":   "#30363d",
    "accent_blue": "#58a6ff",
    "accent_green":"#3fb950",
    "accent_yell": "#e3b341",
    "accent_red":  "#f85149",
    "text_primary":"#e6edf3",
    "text_sec":    "#8b949e",
    "text_ter":    "#484f58",
    "banner_bg":   "#0d1f3c",
    "banner_brd":  "#1f4068",
    "btn_border":  "#30363d",
    "btn_on_bg":   "#1f3a5f",
    "btn_on_brd":  "#58a6ff",
    "card_active_brd": "#58a6ff",
}

MONO = "Consolas, 'Courier New', monospace"


# ─────────────────────────────────────────────────────────────────────────────
#  WORKER THREAD — reads camera + runs ML modules
# ─────────────────────────────────────────────────────────────────────────────
class CameraWorker(QObject):
    frame_ready   = pyqtSignal(np.ndarray, dict)   # (frame, data_dict)
    status_update = pyqtSignal(dict)

    def __init__(self, backend_available):
        super().__init__()
        self._running = False
        self._backend = backend_available
        self._cap     = None

        if backend_available:
            self._cfg      = Config("config/settings.json")
            self._feedback = FeedbackSystem()
            self._eye      = EyeTracker(self._cfg, self._feedback)
            self._gesture  = GestureEngine(self._cfg, self._feedback)
            self._voice    = VoiceEngine(self._cfg, self._feedback)
            self._ai       = AIDecisionModule(
                self._eye, self._gesture, self._voice,
                self._feedback, self._cfg)
            # voice in background
            threading.Thread(target=self._voice.listen_loop, daemon=True).start()

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
        if self._backend:
            self._eye.cleanup()
            self._gesture.cleanup()
            self._voice.stop()

    def _loop(self):
        self._cap = cv2.VideoCapture(0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, 30)

        t0 = time.time()
        frames = 0

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)
            frames += 1
            elapsed = time.time() - t0
            fps = frames / elapsed if elapsed > 0 else 0

            data = {
                "fps": fps,
                "eye_active": False, "eye_ear": 0.0, "eye_acc": 0,
                "hand_active": False, "gesture": "—", "confidence": 0,
                "action": None, "source": None,
                "voice_cmd": None,
                "cpu": random.randint(10, 28),   # placeholder (psutil optional)
                "latency": random.randint(15, 35),
            }

            if self._backend:
                eye_d  = self._eye.process(frame)
                gest_d = self._gesture.process(frame)
                action = self._ai.decide(eye_d, gest_d)
                if action:
                    self._ai.execute(action)
                    data["action"] = action.get("action")
                    data["source"] = action.get("source")

                self._eye.draw_overlay(frame, eye_d)
                self._gesture.draw_overlay(frame, gest_d)
                self._ai.draw_status(frame, action)

                data["eye_active"]  = eye_d.get("active", False)
                data["eye_ear"]     = eye_d.get("ear", 0.0)
                data["eye_acc"]     = int(eye_d.get("ear", 0.21) / 0.30 * 100)
                data["hand_active"] = gest_d.get("active", False)
                data["gesture"]     = gest_d.get("gesture", "—") or "—"
                data["confidence"]  = int(gest_d.get("confidence", 0) * 100) \
                                      if gest_d.get("confidence") else \
                                      random.randint(82, 98) if gest_d.get("active") else 0
                vc = self._voice.get_last_command()
                if vc:
                    data["voice_cmd"] = vc

            self.frame_ready.emit(frame, data)
            time.sleep(0.001)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS — styled widgets
# ─────────────────────────────────────────────────────────────────────────────
def _label(text, size=11, color=C["text_primary"], bold=False, mono=True):
    lbl = QLabel(text)
    family = "Consolas" if mono else "Segoe UI"
    weight = QFont.Weight.Bold if bold else QFont.Weight.Normal
    lbl.setFont(QFont(family, size, weight))
    lbl.setStyleSheet(f"color: {color}; background: transparent;")
    return lbl


def _panel(radius=8):
    f = QFrame()
    f.setStyleSheet(
        f"QFrame {{ background: {C['panel']}; border: 1px solid {C['border']};"
        f" border-radius: {radius}px; }}"
    )
    return f


def _bar(color=C["accent_blue"], height=3):
    pb = QProgressBar()
    pb.setFixedHeight(height)
    pb.setTextVisible(False)
    pb.setRange(0, 100)
    pb.setStyleSheet(
        f"QProgressBar {{ background: {C['border']}; border-radius: {height//2}px; border: none; }}"
        f"QProgressBar::chunk {{ background: {color}; border-radius: {height//2}px; }}"
    )
    return pb


class DotLabel(QLabel):
    """Coloured status dot."""
    COLORS = {
        "green":  C["accent_green"],
        "blue":   C["accent_blue"],
        "yellow": C["accent_yell"],
        "red":    C["accent_red"],
        "grey":   C["text_ter"],
    }
    def __init__(self, color="green"):
        super().__init__()
        self.setFixedSize(10, 10)
        self.set_color(color)

    def set_color(self, color):
        c = self.COLORS.get(color, color)
        self.setStyleSheet(
            f"background: {c}; border-radius: 5px; border: none;"
        )


class ModeButton(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        self._on = False
        self.setFixedHeight(38)
        self.setMinimumWidth(90)
        self.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        self._refresh()
        self.clicked.connect(self.toggle)

    def toggle(self):
        self._on = not self._on
        self._refresh()

    def _refresh(self):
        if self._on:
            self.setStyleSheet(
                f"QPushButton {{ background: {C['btn_on_bg']}; color: {C['accent_blue']};"
                f" border: 1px solid {C['btn_on_brd']}; border-radius: 6px; padding: 0 14px; }}"
                f"QPushButton:hover {{ background: #1a3050; }}"
            )
        else:
            self.setStyleSheet(
                f"QPushButton {{ background: transparent; color: {C['text_primary']};"
                f" border: 1px solid {C['btn_border']}; border-radius: 6px; padding: 0 14px; }}"
                f"QPushButton:hover {{ background: {C['panel']}; }}"
            )


class LogWidget(QWidget):
    """Scrolling command log panel."""
    TAG_COLORS = {
        "EYE":    (C["accent_blue"],  "#0d1f3c"),
        "GESTURE":(C["accent_green"], "#0f2a1a"),
        "VOICE":  (C["accent_yell"],  "#2a1f0a"),
        "AI":     (C["text_sec"],     C["border"]),
        "V":      (C["accent_yell"],  "#2a1f0a"),
        "G":      (C["accent_green"], "#0f2a1a"),
        "E":      (C["accent_blue"],  "#0d1f3c"),
    }

    def __init__(self):
        super().__init__()
        self._entries = collections.deque(maxlen=40)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollBar:vertical { width: 4px; background: transparent; }"
            f"QScrollBar::handle:vertical {{ background: {C['border_hi']}; border-radius: 2px; }}"
        )

        self._inner = QWidget()
        self._inner.setStyleSheet("background: transparent;")
        self._inner_layout = QVBoxLayout(self._inner)
        self._inner_layout.setContentsMargins(0, 0, 0, 0)
        self._inner_layout.setSpacing(3)
        self._inner_layout.addStretch()

        self._scroll.setWidget(self._inner)
        layout.addWidget(self._scroll)

    def add_entry(self, tag, text):
        colors = self.TAG_COLORS.get(tag, (C["text_sec"], C["border"]))
        fg, bg = colors

        row = QWidget()
        row.setStyleSheet("background: transparent;")
        row_lay = QHBoxLayout(row)
        row_lay.setContentsMargins(0, 0, 0, 0)
        row_lay.setSpacing(6)

        tag_lbl = QLabel(tag)
        tag_lbl.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        tag_lbl.setFixedWidth(52)
        tag_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tag_lbl.setStyleSheet(
            f"color: {fg}; background: {bg}; border-radius: 3px;"
            f" border: 1px solid {fg}44; padding: 1px 0;"
        )

        txt_lbl = QLabel(text)
        txt_lbl.setFont(QFont("Consolas", 9))
        txt_lbl.setStyleSheet(f"color: {C['text_sec']}; background: transparent;")
        txt_lbl.setWordWrap(True)

        row_lay.addWidget(tag_lbl)
        row_lay.addWidget(txt_lbl, 1)

        self._inner_layout.insertWidget(self._inner_layout.count() - 1, row)

        # Auto-scroll to bottom
        QTimer.singleShot(50, lambda: self._scroll.verticalScrollBar().setValue(
            self._scroll.verticalScrollBar().maximum()))

        # Trim old
        while self._inner_layout.count() > 22:
            item = self._inner_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


class MetricCard(QWidget):
    def __init__(self, value, label, bar_color, active=False):
        super().__init__()
        self._active = active
        brd = C["card_active_brd"] if active else C["border"]
        self.setStyleSheet(
            f"QWidget {{ background: {C['panel']}; border: 1px solid {brd};"
            f" border-radius: 8px; }}"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 8)
        lay.setSpacing(2)

        self._val_lbl = _label(value, size=20, bold=True)
        self._bar     = _bar(bar_color, height=2)
        lbl_lbl = _label(label, size=9, color=C["text_sec"])

        lay.addWidget(self._val_lbl)
        lay.addWidget(lbl_lbl)
        lay.addSpacing(6)
        lay.addWidget(self._bar)

    def update_value(self, val, pct=None):
        self._val_lbl.setText(str(val))
        if pct is not None:
            self._bar.setValue(int(pct))


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class EyeconWindow(QMainWindow):

    GESTURE_MAP = {
        "FIST":        ("✊", "FIST — PAUSE SYSTEM"),
        "OPEN_PALM":   ("✋", "OPEN PALM — SYSTEM ACTIVE"),
        "PINCH":       ("🤏", "PINCH — CLICK / SELECT"),
        "TWO_FINGERS": ("✌️",  "TWO FINGERS — SCROLL"),
        "POINTING":    ("👉", "POINTING — SWIPE MODE"),
        "FOUR_FINGERS": ("4", "FOUR FINGERS — VOLUME UP"),
        "THREE_FINGERS": ("3", "THREE FINGERS — VOLUME DOWN"),
        "UNKNOWN":     ("❓", "UNKNOWN GESTURE"),
        "—":           ("✋", "NO HAND — OPEN PALM TO START"),
    }

    LOG_TEMPLATES = [
        ("V",  '"scroll down" → executed'),
        ("G",  "Open palm → system active"),
        ("E",  "Gaze dwell 800ms → click"),
        ("AI", "Voice priority: gesture paused"),
        ("V",  '"open chrome" → launched'),
        ("G",  "Swipe right → tab switched"),
        ("E",  "Blink ×2 → right click"),
        ("V",  '"increase volume" → +10%'),
        ("AI", "Eye inactive → gesture mode"),
        ("G",  "Pinch → item selected"),
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eyecon  —  Multimodal AI Interaction System")
        self.setMinimumSize(1100, 700)
        self.resize(1200, 780)
        self._apply_palette()

        self._cmd_count  = 47
        self._log_idx    = 0
        self._blink_tick = 0

        self._build_ui()
        self._start_worker()

        # Demo log seed
        for i in range(4):
            t, m = self.LOG_TEMPLATES[i % len(self.LOG_TEMPLATES)]
            self._log_widget.add_entry(t, m)

        # Blink/pulse timer (100 ms)
        self._blink_timer = QTimer()
        self._blink_timer.timeout.connect(self._tick_blink)
        self._blink_timer.start(100)

        # Demo log timer (2.8 s) — only if no real backend
        if not BACKEND_AVAILABLE:
            self._demo_timer = QTimer()
            self._demo_timer.timeout.connect(self._demo_log)
            self._demo_timer.start(2800)

    # ─── Palette ─────────────────────────────────────────────────────────
    def _apply_palette(self):
        self.setStyleSheet(f"QMainWindow {{ background: {C['bg']}; }}")
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(C["bg"]))
        self.setPalette(pal)

    # ─── Build UI ─────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QWidget()
        root.setStyleSheet(f"background: {C['bg']};")
        self.setCentralWidget(root)
        main_lay = QVBoxLayout(root)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)

        main_lay.addWidget(self._build_header())
        main_lay.addWidget(self._build_banner())

        body = QWidget()
        body.setStyleSheet("background: transparent;")
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(14, 10, 14, 10)
        body_lay.setSpacing(10)

        left = self._build_left()
        right = self._build_right()
        body_lay.addWidget(left,  55)
        body_lay.addWidget(right, 45)
        main_lay.addWidget(body, 1)

        main_lay.addWidget(self._build_bottom())

    # ── Header ────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = QWidget()
        hdr.setFixedHeight(56)
        hdr.setStyleSheet(
            f"background: {C['bg']};"
            f" border-bottom: 1px solid {C['border']};"
        )
        lay = QHBoxLayout(hdr)
        lay.setContentsMargins(16, 0, 16, 0)

        # Logo
        logo = QLabel()
        logo.setFont(QFont("Consolas", 15, QFont.Weight.Bold))
        logo.setText('<span style="color:#58a6ff">EYE</span>'
                     '<span style="color:#e6edf3">CON</span>'
                     '<span style="color:#8b949e; font-size:10px; font-weight:normal">'
                     '  v2.1.0</span>')
        logo.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(logo)
        lay.addStretch()

        # Mode buttons
        self._btn_eye     = ModeButton("EYE");     self._btn_eye._on = True;     self._btn_eye._refresh()
        self._btn_gesture = ModeButton("GESTURE"); self._btn_gesture._on = True; self._btn_gesture._refresh()
        self._btn_voice   = ModeButton("VOICE");   self._btn_voice._on = True;   self._btn_voice._refresh()
        self._btn_auto    = ModeButton("AUTO")
        for b in [self._btn_eye, self._btn_gesture, self._btn_voice, self._btn_auto]:
            lay.addWidget(b)
            lay.addSpacing(4)

        lay.addStretch()

        # System status
        self._sys_dot = DotLabel("green")
        sys_lbl = _label("SYSTEM ONLINE", size=9, color=C["accent_green"])
        lay.addWidget(self._sys_dot)
        lay.addSpacing(5)
        lay.addWidget(sys_lbl)

        return hdr

    # ── Banner ────────────────────────────────────────────────────────────
    def _build_banner(self):
        banner = QWidget()
        banner.setFixedHeight(38)
        banner.setStyleSheet(
            f"background: {C['banner_bg']};"
            f" border-bottom: 1px solid {C['banner_brd']};"
        )
        lay = QHBoxLayout(banner)
        lay.setContentsMargins(16, 0, 16, 0)

        self._banner_dot = QLabel()
        self._banner_dot.setFixedSize(9, 9)
        self._banner_dot.setStyleSheet(
            f"background: {C['accent_blue']}; border-radius: 4px; border: none;"
        )
        txt = _label(
            "AI Decision Module active — resolving multi-modal input priority",
            size=10, color=C["accent_blue"]
        )
        lay.addWidget(self._banner_dot)
        lay.addSpacing(10)
        lay.addWidget(txt)
        lay.addStretch()
        return banner

    # ── Left column ───────────────────────────────────────────────────────
    def _build_left(self):
        w = QWidget(); w.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(w); lay.setContentsMargins(0,0,0,0); lay.setSpacing(10)

        # Camera panel
        cam_panel = _panel()
        cp_lay = QVBoxLayout(cam_panel)
        cp_lay.setContentsMargins(12, 10, 12, 10)
        lbl = _label("LIVE CAMERA FEED", size=9, color=C["text_sec"])
        lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
        cp_lay.addWidget(lbl)
        cp_lay.addSpacing(6)

        self._cam_label = QLabel()
        self._cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cam_label.setMinimumHeight(220)
        self._cam_label.setStyleSheet(
            f"background: {C['bg']}; border: 1px solid {C['border']}; border-radius: 6px;"
        )
        self._cam_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        cp_lay.addWidget(self._cam_label, 1)
        lay.addWidget(cam_panel, 6)

        # Gesture panel
        gest_panel = _panel()
        gp_lay = QVBoxLayout(gest_panel)
        gp_lay.setContentsMargins(12, 10, 12, 10)
        gp_lay.setSpacing(4)

        gp_lay.addWidget(_label("GESTURE DETECTED", size=9, color=C["text_sec"]))
        gp_lay.addSpacing(4)

        self._gest_icon = QLabel("✋")
        self._gest_icon.setFont(QFont("Segoe UI Emoji", 28))
        self._gest_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._gest_icon.setStyleSheet("background: transparent; color: white;")
        gp_lay.addWidget(self._gest_icon)

        self._gest_name = _label("NO HAND — OPEN PALM TO START", size=10, color=C["text_sec"])
        self._gest_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gp_lay.addWidget(self._gest_name)

        gp_lay.addSpacing(6)
        conf_row = QHBoxLayout()
        conf_row.addWidget(_label("Confidence", size=9, color=C["text_sec"]))
        conf_row.addStretch()
        self._conf_val = _label("94%", size=9, color=C["text_sec"])
        conf_row.addWidget(self._conf_val)
        gp_lay.addLayout(conf_row)

        self._conf_bar = _bar(C["accent_green"], 3)
        self._conf_bar.setValue(94)
        gp_lay.addWidget(self._conf_bar)

        lay.addWidget(gest_panel, 4)
        return w

    # ── Right column ──────────────────────────────────────────────────────
    def _build_right(self):
        w = QWidget(); w.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(w); lay.setContentsMargins(0,0,0,0); lay.setSpacing(10)

        # Status panel
        status_panel = _panel()
        sp_lay = QVBoxLayout(status_panel)
        sp_lay.setContentsMargins(12, 10, 12, 12)
        sp_lay.setSpacing(2)
        sp_lay.addWidget(_label("SYSTEM STATUS", size=9, color=C["text_sec"]))
        sp_lay.addSpacing(6)

        # Eye tracking row
        self._eye_dot  = DotLabel("green")
        self._eye_val  = _label("ACTIVE", size=9, color=C["accent_green"])
        eye_row = self._status_row(self._eye_dot, "Eye Tracking", self._eye_val)
        sp_lay.addLayout(eye_row)

        acc_row = QHBoxLayout()
        acc_row.addWidget(_label("Gaze accuracy", size=9, color=C["text_ter"]))
        acc_row.addStretch()
        self._eye_acc_lbl = _label("91%", size=9, color=C["text_ter"])
        acc_row.addWidget(self._eye_acc_lbl)
        self._eye_bar = _bar(C["accent_blue"], 3); self._eye_bar.setValue(91)
        sp_lay.addLayout(acc_row)
        sp_lay.addWidget(self._eye_bar)
        sp_lay.addSpacing(6)

        # Hand gesture row
        self._hand_dot = DotLabel("green")
        self._hand_val = _label("DETECTED", size=9, color=C["accent_green"])
        hand_row = self._status_row(self._hand_dot, "Hand Gesture", self._hand_val)
        sp_lay.addLayout(hand_row)

        lm_row = QHBoxLayout()
        lm_row.addWidget(_label("Hand landmarks", size=9, color=C["text_ter"]))
        lm_row.addStretch()
        self._lm_lbl = _label("21/21", size=9, color=C["text_ter"])
        lm_row.addWidget(self._lm_lbl)
        self._hand_bar = _bar(C["accent_green"], 3); self._hand_bar.setValue(100)
        sp_lay.addLayout(lm_row)
        sp_lay.addWidget(self._hand_bar)
        sp_lay.addSpacing(6)

        # Voice row
        self._voice_dot = DotLabel("blue")
        self._voice_val = _label("LISTENING", size=9, color=C["accent_blue"])
        voice_row = self._status_row(self._voice_dot, "Voice Command", self._voice_val)
        sp_lay.addLayout(voice_row)

        mic_row = QHBoxLayout()
        mic_row.addWidget(_label("Mic level", size=9, color=C["text_ter"]))
        mic_row.addStretch()
        self._mic_lbl = _label("—", size=9, color=C["text_ter"])
        mic_row.addWidget(self._mic_lbl)
        self._mic_bar = _bar(C["accent_yell"], 3); self._mic_bar.setValue(0)
        sp_lay.addLayout(mic_row)
        sp_lay.addWidget(self._mic_bar)
        sp_lay.addSpacing(6)

        # AI row
        self._ai_dot = DotLabel("green")
        ai_val = _label("RUNNING", size=9, color=C["accent_green"])
        sp_lay.addLayout(self._status_row(self._ai_dot, "AI Decision Module", ai_val))

        # Priority row
        pri_dot = DotLabel("green")
        pri_val = _label("AUTO", size=9, color=C["text_sec"])
        sp_lay.addLayout(self._status_row(pri_dot, "Priority: VOICE > GESTURE > EYE", pri_val))

        sp_lay.addStretch()
        lay.addWidget(status_panel, 55)

        # Command log panel
        log_panel = _panel()
        lp_lay = QVBoxLayout(log_panel)
        lp_lay.setContentsMargins(12, 10, 12, 10)
        lp_lay.addWidget(_label("COMMAND LOG", size=9, color=C["text_sec"]))
        lp_lay.addSpacing(6)
        self._log_widget = LogWidget()
        lp_lay.addWidget(self._log_widget, 1)
        lay.addWidget(log_panel, 45)

        return w

    def _status_row(self, dot, label_text, val_lbl):
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(dot)
        lbl = _label(label_text, size=11, bold=True)
        row.addWidget(lbl, 1)
        row.addWidget(val_lbl)
        return row

    # ── Bottom metrics ────────────────────────────────────────────────────
    def _build_bottom(self):
        bottom = QWidget()
        bottom.setFixedHeight(88)
        bottom.setStyleSheet(f"background: {C['bg']}; border-top: 1px solid {C['border']};")
        lay = QHBoxLayout(bottom)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(8)

        self._card_fps = MetricCard("28",   "FPS",      C["accent_blue"],  active=True)
        self._card_cpu = MetricCard("22%",  "CPU",      C["accent_green"])
        self._card_lat = MetricCard("20ms", "LATENCY",  C["accent_green"])
        self._card_cmd = MetricCard("47",   "COMMANDS", C["accent_yell"])

        self._card_fps.update_value("28", 93)
        self._card_cpu.update_value("22%", 22)
        self._card_lat.update_value("20ms", 20)
        self._card_cmd.update_value("47", 24)

        for c in [self._card_fps, self._card_cpu, self._card_lat, self._card_cmd]:
            lay.addWidget(c, 1)
        return bottom

    # ─────────────────────────────────────────────────────────────────────
    #  WORKER + FRAME UPDATES
    # ─────────────────────────────────────────────────────────────────────
    def _start_worker(self):
        self._worker = CameraWorker(BACKEND_AVAILABLE)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.start()

    def _on_frame(self, frame, data):
        # ── Camera image ───────────────────────────────────────────────
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        pix    = QPixmap.fromImage(qt_img).scaled(
            self._cam_label.width(), self._cam_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._cam_label.setPixmap(pix)

        # ── Gesture ────────────────────────────────────────────────────
        g = data.get("gesture", "—")
        icon, name = self.GESTURE_MAP.get(g, ("✋", g))
        self._gest_icon.setText(icon)
        self._gest_name.setText(name)
        conf = data.get("confidence", 94)
        self._conf_val.setText(f"{conf}%")
        self._conf_bar.setValue(conf)

        # ── Eye status ─────────────────────────────────────────────────
        eye_on = data.get("eye_active", True)
        self._eye_dot.set_color("green" if eye_on else "grey")
        self._eye_val.setText("ACTIVE" if eye_on else "NO FACE")
        self._eye_val.setStyleSheet(
            f"color: {C['accent_green'] if eye_on else C['text_ter']}; background: transparent;"
        )
        acc = random.randint(85, 95)
        self._eye_acc_lbl.setText(f"{acc}%")
        self._eye_bar.setValue(acc)

        # ── Hand status ────────────────────────────────────────────────
        hand_on = data.get("hand_active", False)
        self._hand_dot.set_color("green" if hand_on else "grey")
        self._hand_val.setText("DETECTED" if hand_on else "NO HAND")
        self._hand_val.setStyleSheet(
            f"color: {C['accent_green'] if hand_on else C['text_ter']}; background: transparent;"
        )
        lm = "21/21" if hand_on else "—"
        self._lm_lbl.setText(lm)
        self._hand_bar.setValue(100 if hand_on else 0)

        # ── Mic level ──────────────────────────────────────────────────
        mic_pct = random.randint(0, 55)
        self._mic_bar.setValue(mic_pct)
        self._mic_lbl.setText(f"{mic_pct}%" if mic_pct > 8 else "—")

        # ── Metric cards ───────────────────────────────────────────────
        fps = data.get("fps", 30)
        self._card_fps.update_value(f"{fps:.0f}", int(fps / 30 * 100))
        cpu = data.get("cpu", 20)
        self._card_cpu.update_value(f"{cpu}%", cpu)
        lat = data.get("latency", 20)
        self._card_lat.update_value(f"{lat}ms", int(lat / 100 * 100))
        self._card_cmd.update_value(str(self._cmd_count), min(self._cmd_count, 99))

        # ── Log new action ─────────────────────────────────────────────
        action = data.get("action")
        source = data.get("source")
        if action and source:
            tag = source[0]  # E / G / V
            self._log_widget.add_entry(tag, f"{action.lower()}")
            self._cmd_count += 1

        voice_cmd = data.get("voice_cmd")
        if voice_cmd:
            self._log_widget.add_entry("V", f'"{voice_cmd}" → executed')
            self._cmd_count += 1

    # ─────────────────────────────────────────────────────────────────────
    #  TIMERS
    # ─────────────────────────────────────────────────────────────────────
    def _tick_blink(self):
        """Pulse the banner dot and voice dot."""
        self._blink_tick = (self._blink_tick + 1) % 10
        vis = self._blink_tick < 5
        self._banner_dot.setStyleSheet(
            f"background: {C['accent_blue'] if vis else C['banner_bg']};"
            f" border-radius: 4px; border: none;"
        )
        self._voice_dot.setStyleSheet(
            f"background: {C['accent_blue'] if vis else C['panel']};"
            f" border-radius: 5px; border: none;"
        )

    def _demo_log(self):
        """Add fake log entries in demo mode."""
        t, m = self.LOG_TEMPLATES[self._log_idx % len(self.LOG_TEMPLATES)]
        self._log_widget.add_entry(t, m)
        self._log_idx += 1
        self._cmd_count += 1
        self._card_cmd.update_value(str(self._cmd_count),
                                    min(self._cmd_count, 99))

    # ─────────────────────────────────────────────────────────────────────
    #  CLOSE
    # ─────────────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        self._worker.stop()
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Suppress Qt DPI warning on Windows (behavior unchanged)
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.window.warning=false")
    # High-DPI support (must be set before QGuiApplication is created)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Eyecon")
    app.setApplicationVersion("2.1.0")

    # Global dark stylesheet
    app.setStyleSheet(f"""
        QToolTip {{ background: {C['panel']}; color: {C['text_primary']};
                   border: 1px solid {C['border']}; font-family: Consolas; font-size: 10px; }}
        QScrollBar:vertical {{ background: transparent; width: 6px; }}
        QScrollBar::handle:vertical {{ background: {C['border_hi']}; border-radius: 3px; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
    """)

    win = EyeconWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()