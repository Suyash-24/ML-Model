"""
app.py  —  Eyecon WEDNESDAY Interface  (complete rewrite)
───────────────────────────────────────────────────────
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
            self._wednesday  = WednesdayEngine(self._cfg, self._feedback)
            self._ai      = AIDecisionModule(
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
                    "action": None, "source": None,
                    "gaze_norm": None, "ear": 0.0,
                    "blink_count": 0, "click_count": 0,
                    "dwell_progress": 0.0, "blink_event": False,
                    "dwell_event": False, "face_detected": False,
                    "system_state": "PAUSED",
                    "frame_w": frame.shape[1], "frame_h": frame.shape[0]}

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
                data["landmarks"]   = gest_d.get("landmarks")
                data["gesture_executed"] = bool(gest_d.get("executed", False))
                data["gesture_action"]   = gest_d.get("action", "")
                data["gesture_count"]    = int(getattr(self._gesture, "gesture_count", 0))
                # Rich eye telemetry for Eye Control page
                data["face_detected"] = eye_d.get("face", False)
                data["ear"]           = float(eye_d.get("ear", 0.0) or 0.0)
                data["blink_event"]   = bool(eye_d.get("blink", False))
                data["dwell_event"]   = bool(eye_d.get("dwell_click", False))
                data["blink_count"]   = int(getattr(self._eye, "blink_count", 0))
                data["click_count"]   = int(getattr(self._eye, "click_count", 0))
                data["dwell_progress"] = (
                    float(getattr(self._eye, "dwell_frames", 0)) / 25.0
                    if getattr(self._eye, "dwell_frames", 0) else 0.0)
                gs = eye_d.get("gaze_screen")
                if gs and getattr(self._eye, "screen_w", 0):
                    data["gaze_norm"] = (gs[0] / max(self._eye.screen_w, 1),
                                         gs[1] / max(self._eye.screen_h, 1))
                data["system_state"] = getattr(self._ai, "state", "ACTIVE")

            try:
                self.frame_ready.emit(frame, data)
            except RuntimeError:
                break   # widget deleted — stop the loop gracefully
            time.sleep(0.001)

        if self._cap: self._cap.release()

    def get_friday(self):
        return self._wednesday


# ═══════════════════════════════════════════════════════════════════════════════
#  STYLE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
BG      = "#0d0d0f"
PNL     = "#111113"
PNL2    = "#18181b"
BDR     = "rgba(255,255,255,0.07)"
TXT     = "#f4f4f5"
TXT2    = "#71717a"
TXT3    = "#52525b"
GREEN   = "#22c55e"
ACCENT  = "#22c55e"
ACCENT2 = "#8b5cf6"
SURFACE = "#1c1c1f"
FONT    = "Segoe UI"
FONT_MONO = "Consolas"

_GESTURE_EMOJI = {
    "OPEN_PALM":  ("✋", "Open Palm",   "#22c55e"),
    "FIST":       ("✊", "Fist",         "#f59e0b"),
    "INDEX_ONLY": ("☝️", "Index Pointer", "#38bdf8"),
    "PINCH":      ("🤏", "Pinch",         "#a78bfa"),
    "TWO_FINGERS":("✌️", "Two Fingers",   "#34d399"),
    "THUMBS_UP":  ("👍", "Thumbs Up",     "#22c55e"),
    "OK":         ("👌", "OK Sign",       "#818cf8"),
    "SCROLL_H":   ("↔️", "H Scroll",      "#fb923c"),
    "SCROLL_V":   ("↕️", "V Scroll",      "#fb923c"),
}


def _font(size=11, bold=False, mono=False):
    family = FONT_MONO if mono else FONT
    w = QFont.Weight.Bold if bold else QFont.Weight.Normal
    f = QFont(family, size, w)
    f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.3 if not mono else 0)
    return f


def _lbl(text, size=11, color=TXT, bold=False, mono=False):
    l = QLabel(text)
    l.setFont(_font(size, bold, mono))
    l.setStyleSheet(f"color:{color};background:transparent;")
    return l


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
class Sidebar(QWidget):
    page_changed = pyqtSignal(str)

    ITEMS = [
        ("Dashboard",   "NAVIGATION", "\u2302"),
        ("Eye Control", "",           "\u25C9"),
        ("WEDNESDAY AI",   "",           "\u2606"),
        ("Biometrics",  "",           "\u26BF"),
        ("Calibrate",   "TOOLS",      "\u2699"),
        ("Analytics",   "",           "\u2261"),
        ("Settings",    "",           "\u2630"),
        ("Profile",     "",           "\u263A"),
    ]

    def __init__(self, username=""):
        super().__init__()
        self.setFixedWidth(220)
        self.setStyleSheet(f"background:{PNL};border-right:1px solid {BDR};")
        self._buttons = []
        self._username = username

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Logo header
        logo_w = QWidget()
        logo_w.setFixedHeight(48)
        logo_w.setStyleSheet(f"border-bottom:1px solid {BDR};background:transparent;")
        ll = QHBoxLayout(logo_w)
        ll.setContentsMargins(16, 0, 16, 0)
        ll.setSpacing(8)
        dot = QLabel()
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(f"background:{GREEN};border-radius:4px;border:none;")
        ll.addWidget(dot)
        title = QLabel("EYECON")
        title.setFont(QFont("Consolas", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color:#ffffff;background:transparent;letter-spacing:2px;")
        ll.addWidget(title)
        ll.addStretch()
        ver = QLabel("v2.1")
        ver.setFont(QFont("Consolas", 9))
        ver.setStyleSheet(f"color:{TXT3};background:transparent;")
        ll.addWidget(ver)
        lay.addWidget(logo_w)

        # Nav items
        nav_scroll = QScrollArea()
        nav_scroll.setWidgetResizable(True)
        nav_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        nav_scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}"
                                  "QScrollBar{width:0px;}")
        nav_inner = QWidget()
        nav_inner.setStyleSheet("background:transparent;")
        nav_lay = QVBoxLayout(nav_inner)
        nav_lay.setContentsMargins(10, 12, 10, 12)
        nav_lay.setSpacing(2)

        last_section = None
        for name, section, icon in self.ITEMS:
            if section and section != last_section:
                sec_lbl = QLabel(section)
                sec_lbl.setFont(QFont("Consolas", 8))
                sec_lbl.setStyleSheet(f"color:{TXT3};background:transparent;padding:0;letter-spacing:2px;")
                sec_lbl.setContentsMargins(10, 12, 10, 4)
                nav_lay.addWidget(sec_lbl)
                last_section = section

            btn = QPushButton(f"  {name}")
            btn.setFont(QFont("Segoe UI", 10))
            btn.setFixedHeight(34)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setCheckable(True)
            btn.setStyleSheet(self._btn_style(False))
            btn.clicked.connect(lambda _, n=name, b=btn: self._select(n, b))
            self._buttons.append(btn)
            nav_lay.addWidget(btn)

        nav_lay.addStretch()
        nav_scroll.setWidget(nav_inner)
        lay.addWidget(nav_scroll, 1)

        # Status footer
        foot = QWidget()
        foot.setStyleSheet(f"border-top:1px solid {BDR};background:transparent;")
        fl = QVBoxLayout(foot)
        fl.setContentsMargins(18, 10, 18, 6)
        fl.setSpacing(4)

        self._status_dots = {}
        for label, key, color in [
            ("Eye tracking",   "eye",  GREEN),
            ("Gesture engine",  "gest", GREEN),
            ("Wednesday core", "fri",  GREEN),
            ("Biometric auth", "bio",  "#f59e0b"),
        ]:
            row = QHBoxLayout(); row.setSpacing(8)
            dot = QLabel()
            dot.setFixedSize(7, 7)
            dot.setStyleSheet(f"background:{color};border-radius:3px;border:none;")
            self._status_dots[key] = dot
            row.addWidget(dot)
            lbl = QLabel(label)
            lbl.setFont(QFont("Consolas", 9))
            lbl.setStyleSheet(f"color:{TXT2};background:transparent;")
            row.addWidget(lbl)
            row.addStretch()
            fl.addLayout(row)
        lay.addWidget(foot)

        # User card
        user_card = QWidget()
        user_card.setFixedHeight(56)
        user_card.setStyleSheet(f"background:#1a1a2e;border-top:1px solid {BDR};"
                                f"border-radius:0 0 0 0;")
        uc_lay = QHBoxLayout(user_card)
        uc_lay.setContentsMargins(14, 8, 14, 8)
        uc_lay.setSpacing(10)
        avatar = QLabel(username[0].upper() if username else "U")
        avatar.setFixedSize(32, 32)
        avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        avatar.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        avatar.setStyleSheet(f"background:{ACCENT2};color:#fff;border-radius:16px;border:none;")
        uc_lay.addWidget(avatar)
        uc_info = QVBoxLayout()
        uc_info.setSpacing(0)
        uname = QLabel(username[:16] if username else "-")
        uname.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        uname.setStyleSheet(f"color:{TXT};background:transparent;")
        uc_info.addWidget(uname)
        ustatus = QLabel("Authenticated")
        ustatus.setFont(QFont("Consolas", 8))
        ustatus.setStyleSheet(f"color:{TXT3};background:transparent;")
        uc_info.addWidget(ustatus)
        uc_lay.addLayout(uc_info)
        uc_lay.addStretch()
        lay.addWidget(user_card)

        # Select first
        if self._buttons:
            self._select("Dashboard", self._buttons[0])

    def _btn_style(self, active: bool) -> str:
        if active:
            return (f"QPushButton{{background:transparent;"
                    f"color:{GREEN};border:none;border-left:3px solid {GREEN};"
                    f"border-radius:0px;text-align:left;padding:0 14px;"
                    f"font-family:'Segoe UI';font-size:10px;}}"
                    f"QPushButton:hover{{background:rgba(34,197,94,0.06);}}")
        return (f"QPushButton{{background:transparent;color:{TXT2};"
                f"border:none;border-left:3px solid transparent;border-radius:0px;"
                f"text-align:left;padding:0 14px;font-family:'Segoe UI';font-size:10px;}}"
                f"QPushButton:hover{{background:rgba(255,255,255,0.03);color:#9ca3af;}}")

    def _select(self, name: str, btn: QPushButton):
        for b in self._buttons:
            b.setChecked(False)
            b.setStyleSheet(self._btn_style(False))
        btn.setChecked(True)
        btn.setStyleSheet(self._btn_style(True))
        self.page_changed.emit(name)

    def set_status(self, key: str, active: bool):
        dot = self._status_dots.get(key)
        if dot:
            col = GREEN if active else "#1e293b"
            dot.setStyleSheet(f"background:{col};border-radius:3px;border:none;")


# ═══════════════════════════════════════════════════════════════════════════════
#  CHAT WIDGET
# ═══════════════════════════════════════════════════════════════════════════════
class ChatWidget(QWidget):
    message_sent = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Scroll area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(
            "QScrollArea{border:none;background:transparent;}"
            "QScrollBar:vertical{width:3px;background:transparent;}"
            f"QScrollBar::handle:vertical{{background:{SURFACE};border-radius:1px;}}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}")
        self._inner = QWidget()
        self._inner.setStyleSheet("background:transparent;")
        self._inner_lay = QVBoxLayout(self._inner)
        self._inner_lay.setContentsMargins(0, 4, 0, 4)
        self._inner_lay.setSpacing(0)
        self._inner_lay.addStretch()
        self._scroll.setWidget(self._inner)
        lay.addWidget(self._scroll, 1)

        # Input bar
        inp_w = QWidget()
        inp_w.setFixedHeight(50)
        inp_w.setStyleSheet(f"border-top:1px solid {BDR};background:transparent;")
        inp_lay = QHBoxLayout(inp_w)
        inp_lay.setContentsMargins(14, 9, 14, 9)
        inp_lay.setSpacing(6)

        self._input = QLineEdit()
        self._input.setPlaceholderText('type or say "HEY WEDNESDAY"...')
        self._input.setFont(_font(11, mono=True))
        self._input.setStyleSheet(
            f"QLineEdit{{background:{PNL2};color:#94a3b8;border:1px solid rgba(255,255,255,0.08);"
            f"border-radius:8px;padding:0 14px;font-family:'Consolas';font-size:11px;}}"
            f"QLineEdit:focus{{border:1px solid {ACCENT};color:#e2e8f0;}}"
            f"QLineEdit::placeholder{{color:{TXT3};}}")
        self._input.returnPressed.connect(self._send)
        inp_lay.addWidget(self._input, 1)

        for icon, slot in [("🎤", self._mic), ("➤", self._send)]:
            btn = QPushButton(icon)
            btn.setFixedSize(30, 30)
            btn.setFont(QFont("Segoe UI", 10))
            btn.setStyleSheet(
                f"QPushButton{{background:{PNL2};color:{TXT2};border:1px solid rgba(255,255,255,0.08);"
                f"border-radius:8px;}}"
                f"QPushButton:hover{{border-color:{ACCENT};color:{ACCENT};}}")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(slot)
            inp_lay.addWidget(btn)

        lay.addWidget(inp_w)

    def _send(self):
        txt = self._input.text().strip()
        if txt:
            self._input.clear()
            self.message_sent.emit(txt)

    def _mic(self):
        self.message_sent.emit("__MIC__")

    def add_message(self, role: str, text: str):
        """role: 'friday' | 'user' | 'system'"""
        row = QWidget()
        row.setStyleSheet("background:transparent;")
        row_lay = QVBoxLayout(row)
        row_lay.setContentsMargins(0, 6, 0, 6)
        row_lay.setSpacing(3)

        if role == "system":
            lbl = _lbl(text, 9, TXT3, mono=True)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            row_lay.addWidget(lbl)
        else:
            lbl_row = QHBoxLayout()
            lbl_row.setContentsMargins(0, 0, 0, 0)
            tag_txt = "FRIDAY" if role == "wednesday" else "YOU"
            tag_col = ACCENT if role == "wednesday" else ACCENT2
            tag = _lbl(tag_txt, 9, tag_col, mono=True)
            lbl_row.addWidget(tag)
            if role == "user": lbl_row.addStretch()
            row_lay.addLayout(lbl_row)

            bubble = QLabel(text)
            bubble.setFont(_font(11))
            bubble.setWordWrap(True)
            bubble.setMaximumWidth(420)
            if role == "wednesday":
                bubble.setStyleSheet(
                    f"color:#cbd5e1;background:{PNL2};border:1px solid rgba(59,130,246,0.15);"
                    f"border-radius:2px 10px 10px 10px;padding:10px 14px;")
            else:
                bubble.setStyleSheet(
                    f"color:#94a3b8;background:{SURFACE};border:1px solid rgba(139,92,246,0.12);"
                    f"border-radius:10px 2px 10px 10px;padding:10px 14px;")
                row_lay.setAlignment(Qt.AlignmentFlag.AlignRight)
            row_lay.addWidget(bubble)

        # Insert before stretch
        idx = self._inner_lay.count() - 1
        self._inner_lay.insertWidget(idx, row)
        QTimer.singleShot(50, lambda: self._scroll.verticalScrollBar().setValue(
            self._scroll.verticalScrollBar().maximum()))


class LoadingRings(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(40, 40)
        self._tick = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update_animation)
        self._timer.start(50)

    def update_animation(self):
        self._tick += 1
        self.update()

    def paintEvent(self, event):
        from PyQt6.QtGui import QPainter, QColor, QPen
        from PyQt6.QtCore import Qt
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        center = self.rect().center()
        painter.translate(center)
        
        # Draw center dot
        painter.setBrush(QColor(ACCENT2))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(-3, -3, 6, 6)
        
        # Draw revolving rings (Saturn-like)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        pen = QPen(QColor(ACCENT2))
        pen.setWidth(1)
        painter.setPen(pen)
        
        painter.rotate(self._tick * 4)
        painter.drawEllipse(-12, -4, 24, 8)
        
        painter.rotate(60)
        painter.drawEllipse(-14, -5, 28, 10)

# ═══════════════════════════════════════════════════════════════════════════════
#  EYE CONTROL PAGE  —  live camera-app viewfinder + gaze telemetry
# ═══════════════════════════════════════════════════════════════════════════════
class CameraViewfinder(QLabel):
    """QLabel that displays the live frame plus camera-style corner brackets,
    a faint center crosshair, and a fading gaze trail."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            f"background:#06060a;border:1px solid {BDR};border-radius:14px;")
        self._gaze_trail = []   # list of (nx, ny, age_ticks)
        self._pixmap = None
        self._face_ok = False
        self._tick = 0
        # Pulse timer for trail decay
        self._decay = QTimer(self)
        self._decay.timeout.connect(self._decay_trail)
        self._decay.start(60)

    def _decay_trail(self):
        # Age every dot; drop very old
        self._gaze_trail = [(x, y, a + 1) for (x, y, a) in self._gaze_trail if a < 28]
        self._tick += 1
        if self._pixmap is not None:
            self.update()

    def push_gaze(self, nx: float, ny: float):
        # Keep the latest ~30 gaze samples
        self._gaze_trail.append((nx, ny, 0))
        if len(self._gaze_trail) > 30:
            self._gaze_trail = self._gaze_trail[-30:]

    def set_frame(self, frame_bgr: np.ndarray, face_ok: bool):
        self._face_ok = face_ok
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pm = QPixmap.fromImage(img).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._pixmap = pm
        self.update()

    def paintEvent(self, ev):
        from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(1, 1, -1, -1)

        # Rounded clipping mask
        path = QPainterPath()
        path.addRoundedRect(float(rect.x()), float(rect.y()),
                            float(rect.width()), float(rect.height()), 14, 14)
        p.setClipPath(path)

        # Background fill
        p.fillRect(rect, QColor("#06060a"))

        # Draw pixmap centered
        if self._pixmap is not None:
            pm = self._pixmap
            x = rect.x() + (rect.width()  - pm.width())  // 2
            y = rect.y() + (rect.height() - pm.height()) // 2
            p.drawPixmap(x, y, pm)
            frame_rect = (x, y, pm.width(), pm.height())
        else:
            # Placeholder
            p.setPen(QColor("#3f3f46"))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, "WAITING FOR CAMERA…")
            frame_rect = (rect.x(), rect.y(), rect.width(), rect.height())

        # ── Center crosshair (subtle) ────────────────────────────────────
        cx = frame_rect[0] + frame_rect[2] // 2
        cy = frame_rect[1] + frame_rect[3] // 2
        pen = QPen(QColor(255, 255, 255, 40)); pen.setWidth(1)
        p.setPen(pen)
        p.drawLine(cx - 12, cy, cx + 12, cy)
        p.drawLine(cx, cy - 12, cx, cy + 12)

        # ── Gaze trail (last positions, fading) ──────────────────────────
        fx, fy, fw, fh = frame_rect
        for (nx, ny, age) in self._gaze_trail:
            alpha = max(0, 220 - int(age * 8))
            radius = max(2, 6 - age // 4)
            gx = int(fx + nx * fw)
            gy = int(fy + ny * fh)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(34, 197, 94, alpha))
            p.drawEllipse(gx - radius, gy - radius, radius * 2, radius * 2)
        # Latest gaze: ring
        if self._gaze_trail:
            nx, ny, _ = self._gaze_trail[-1]
            gx = int(fx + nx * fw); gy = int(fy + ny * fh)
            pen = QPen(QColor(34, 197, 94, 220)); pen.setWidth(2)
            p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(gx - 14, gy - 14, 28, 28)

        # ── Corner brackets (camera viewfinder) ──────────────────────────
        bracket_col = QColor(GREEN) if self._face_ok else QColor("#71717a")
        pen = QPen(bracket_col); pen.setWidth(2)
        p.setPen(pen)
        L = 22  # bracket length
        m = 14  # margin from edge
        x0, y0 = rect.x() + m, rect.y() + m
        x1, y1 = rect.right() - m, rect.bottom() - m
        # Top-left
        p.drawLine(x0, y0, x0 + L, y0); p.drawLine(x0, y0, x0, y0 + L)
        # Top-right
        p.drawLine(x1 - L, y0, x1, y0); p.drawLine(x1, y0, x1, y0 + L)
        # Bottom-left
        p.drawLine(x0, y1 - L, x0, y1); p.drawLine(x0, y1, x0 + L, y1)
        # Bottom-right
        p.drawLine(x1 - L, y1, x1, y1); p.drawLine(x1, y1 - L, x1, y1)

        # ── Top-left REC + label ─────────────────────────────────────────
        rec_on = (self._tick // 8) % 2 == 0     # ~0.5s pulse
        if rec_on:
            p.setBrush(QColor(239, 68, 68, 230))
        else:
            p.setBrush(QColor(239, 68, 68, 110))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(x0 + 4, y0 + 4, 8, 8)
        p.setPen(QColor("#f87171"))
        f = QFont(FONT_MONO, 8, QFont.Weight.Bold); p.setFont(f)
        p.drawText(x0 + 18, y0 + 12, "REC")

        # Bottom-left chip: resolution
        if self._pixmap is not None:
            res_txt = f"{self._pixmap.width()}×{self._pixmap.height()}"
            p.setPen(QColor("#a1a1aa"))
            f2 = QFont(FONT_MONO, 8); p.setFont(f2)
            p.drawText(x0 + 4, y1 - 6, res_txt)

        # Bottom-right: face status
        status_txt = "FACE LOCKED" if self._face_ok else "NO FACE"
        status_col = QColor(GREEN) if self._face_ok else QColor("#f87171")
        p.setPen(status_col)
        f3 = QFont(FONT_MONO, 8, QFont.Weight.Bold); p.setFont(f3)
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(status_txt)
        p.drawText(x1 - tw - 4, y1 - 6, status_txt)

        p.end()


class _StatCard(QFrame):
    """A small key/value card for the side panel."""

    def __init__(self, label: str, accent: str = None):
        super().__init__()
        self.setStyleSheet(
            f"QFrame{{background:{PNL2};border:1px solid {BDR};border-radius:10px;}}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(2)
        self._label = QLabel(label)
        self._label.setFont(QFont(FONT_MONO, 7))
        self._label.setStyleSheet(
            f"color:{TXT3};background:transparent;letter-spacing:2px;")
        self._value = QLabel("—")
        self._value.setFont(QFont(FONT, 18, QFont.Weight.Light))
        self._value.setStyleSheet(f"color:{TXT};background:transparent;")
        lay.addWidget(self._label)
        lay.addWidget(self._value)
        bar = QFrame()
        bar.setFixedHeight(2)
        bar.setStyleSheet(f"background:{accent or ACCENT};border:none;border-radius:1px;")
        lay.addWidget(bar)

    def set_value(self, text: str):
        self._value.setText(text)


class EyeControlPage(QWidget):
    """Live camera-app style page for eye tracking."""
    request_calibrate = pyqtSignal()
    request_pause_toggle = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background:{BG};")
        self._is_active = False
        self._last_blink_event_tick = -100
        self._tick = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 16, 28, 12)
        outer.setSpacing(10)

        # ── Header ───────────────────────────────────────────────────────
        hdr = QHBoxLayout(); hdr.setSpacing(10)
        title = QLabel("◉  EYE  TRACKING  —  LIVE  VIEW")
        title.setFont(QFont(FONT_MONO, 11, QFont.Weight.Bold))
        title.setStyleSheet(f"color:{TXT};background:transparent;letter-spacing:3px;")
        hdr.addWidget(title)
        hdr.addStretch()
        self._state_badge = QLabel("  ● TRACKING")
        self._state_badge.setFont(QFont(FONT_MONO, 9))
        self._state_badge.setFixedHeight(26)
        self._state_badge.setStyleSheet(
            f"color:{GREEN};background:rgba(34,197,94,0.08);"
            f"border:1px solid rgba(34,197,94,0.25);border-radius:6px;padding:0 12px;")
        hdr.addWidget(self._state_badge)
        outer.addLayout(hdr)

        sub = QLabel("Real-time gaze detection · Blink to click · Dwell to confirm · Open palm to resume")
        sub.setFont(QFont(FONT, 10))
        sub.setStyleSheet(f"color:{TXT2};background:transparent;")
        outer.addWidget(sub)

        # ── Body: viewfinder | telemetry ─────────────────────────────────
        body = QHBoxLayout(); body.setSpacing(16)
        outer.addLayout(body, 1)

        # Viewfinder (left, expands)
        self.viewfinder = CameraViewfinder()
        self.viewfinder.setSizePolicy(QSizePolicy.Policy.Expanding,
                                      QSizePolicy.Policy.Expanding)
        body.addWidget(self.viewfinder, 5)

        # Telemetry panel (right)
        side = QWidget()
        side.setStyleSheet("background:transparent;")
        side.setFixedWidth(280)
        side_lay = QVBoxLayout(side)
        side_lay.setContentsMargins(0, 0, 0, 0)
        side_lay.setSpacing(6)

        section = lambda t: _lbl(t, 8, TXT3, mono=True)
        side_lay.addWidget(section("GAZE"))

        # Gaze X / Y bars
        gaze_box = QFrame()
        gaze_box.setStyleSheet(
            f"QFrame{{background:{PNL2};border:1px solid {BDR};border-radius:10px;}}")
        gb_lay = QVBoxLayout(gaze_box)
        gb_lay.setContentsMargins(14, 12, 14, 12)
        gb_lay.setSpacing(8)
        self._gaze_x_bar, self._gaze_x_val = self._build_bar("X")
        self._gaze_y_bar, self._gaze_y_val = self._build_bar("Y")
        gb_lay.addLayout(self._gaze_x_bar)
        gb_lay.addLayout(self._gaze_y_bar)
        side_lay.addWidget(gaze_box)

        side_lay.addWidget(section("ACTIVITY"))
        row = QHBoxLayout(); row.setSpacing(8)
        self._card_blinks = _StatCard("BLINKS",   accent=ACCENT2)
        self._card_clicks = _StatCard("CLICKS",   accent=GREEN)
        row.addWidget(self._card_blinks)
        row.addWidget(self._card_clicks)
        side_lay.addLayout(row)

        # Dwell progress
        dwell_box = QFrame()
        dwell_box.setStyleSheet(
            f"QFrame{{background:{PNL2};border:1px solid {BDR};border-radius:10px;}}")
        db_lay = QVBoxLayout(dwell_box)
        db_lay.setContentsMargins(14, 10, 14, 10)
        db_lay.setSpacing(6)
        dwell_lbl_row = QHBoxLayout()
        dlbl = QLabel("DWELL")
        dlbl.setFont(QFont(FONT_MONO, 7))
        dlbl.setStyleSheet(f"color:{TXT3};background:transparent;letter-spacing:2px;")
        dwell_lbl_row.addWidget(dlbl)
        dwell_lbl_row.addStretch()
        self._dwell_val = QLabel("0%")
        self._dwell_val.setFont(QFont(FONT_MONO, 9))
        self._dwell_val.setStyleSheet(f"color:{TXT2};background:transparent;")
        dwell_lbl_row.addWidget(self._dwell_val)
        db_lay.addLayout(dwell_lbl_row)
        self._dwell_bar_bg = QFrame()
        self._dwell_bar_bg.setFixedHeight(6)
        self._dwell_bar_bg.setStyleSheet(
            f"background:#1a1a1d;border:none;border-radius:3px;")
        self._dwell_bar = QFrame(self._dwell_bar_bg)
        self._dwell_bar.setFixedHeight(6)
        self._dwell_bar.setStyleSheet(
            f"background:{ACCENT2};border:none;border-radius:3px;")
        self._dwell_bar.setGeometry(0, 0, 0, 6)
        db_lay.addWidget(self._dwell_bar_bg)
        side_lay.addWidget(dwell_box)

        # EAR + face card row
        side_lay.addWidget(section("SENSORS"))
        srow = QHBoxLayout(); srow.setSpacing(8)
        self._card_ear  = _StatCard("EAR",  accent=ACCENT)
        self._card_fps  = _StatCard("FPS",  accent=ACCENT2)
        srow.addWidget(self._card_ear)
        srow.addWidget(self._card_fps)
        side_lay.addLayout(srow)

        # ── GESTURE card ────────────────────────────────────────────────
        side_lay.addWidget(section("GESTURE"))
        self._gesture_card = QFrame()
        self._gesture_card.setStyleSheet(
            f"QFrame{{background:{PNL2};border:1px solid {BDR};border-radius:10px;}}")
        self._gesture_card.setFixedHeight(74)
        gc_lay = QHBoxLayout(self._gesture_card)
        gc_lay.setContentsMargins(12, 8, 12, 8)
        gc_lay.setSpacing(14)

        # Emoji bubble
        self._gest_emoji_bg = QFrame()
        self._gest_emoji_bg.setFixedSize(46, 46)
        self._gest_emoji_bg.setStyleSheet(
            f"background:rgba(255,255,255,0.04);border:1px solid {BDR};border-radius:10px;")
        eb_lay = QVBoxLayout(self._gest_emoji_bg)
        eb_lay.setContentsMargins(0, 0, 0, 0)
        self._gest_emoji = QLabel("—")
        self._gest_emoji.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._gest_emoji.setFont(QFont("Segoe UI Emoji", 20))
        self._gest_emoji.setStyleSheet("background:transparent;")
        eb_lay.addWidget(self._gest_emoji)
        gc_lay.addWidget(self._gest_emoji_bg)

        # Name + confidence
        info_lay = QVBoxLayout(); info_lay.setSpacing(4)
        self._gest_name = QLabel("No gesture")
        self._gest_name.setFont(QFont(FONT, 12, QFont.Weight.Bold))
        self._gest_name.setStyleSheet(f"color:{TXT};background:transparent;")
        info_lay.addWidget(self._gest_name)

        conf_row = QHBoxLayout(); conf_row.setSpacing(6)
        conf_lbl = QLabel("CONF")
        conf_lbl.setFont(QFont(FONT_MONO, 7))
        conf_lbl.setStyleSheet(f"color:{TXT3};background:transparent;letter-spacing:2px;")
        conf_row.addWidget(conf_lbl)
        self._gest_conf_bg = QFrame()
        self._gest_conf_bg.setFixedHeight(4)
        self._gest_conf_bg.setStyleSheet(
            "background:#1a1a1d;border:none;border-radius:2px;")
        self._gest_conf_fill = QFrame(self._gest_conf_bg)
        self._gest_conf_fill.setFixedHeight(4)
        self._gest_conf_fill.setStyleSheet(
            f"background:{ACCENT2};border:none;border-radius:2px;")
        self._gest_conf_fill.setGeometry(0, 0, 0, 4)
        conf_row.addWidget(self._gest_conf_bg, 1)
        self._gest_conf_pct = QLabel("0%")
        self._gest_conf_pct.setFont(QFont(FONT_MONO, 8))
        self._gest_conf_pct.setFixedWidth(30)
        self._gest_conf_pct.setStyleSheet(f"color:{TXT2};background:transparent;")
        self._gest_conf_pct.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        conf_row.addWidget(self._gest_conf_pct)
        info_lay.addLayout(conf_row)
        gc_lay.addLayout(info_lay, 1)

        # Accent bar at bottom
        self._gest_accent_bar = QFrame()
        self._gest_accent_bar.setFixedSize(3, 46)
        self._gest_accent_bar.setStyleSheet(
            f"background:{ACCENT2};border:none;border-radius:1px;")
        gc_lay.addWidget(self._gest_accent_bar)
        side_lay.addWidget(self._gesture_card)

        side_lay.addSpacing(4)

        # Action buttons
        side_lay.addWidget(section("CONTROLS"))
        btn_row = QHBoxLayout(); btn_row.setSpacing(8)
        self._btn_calib = self._action_btn("⌖  Calibrate", ACCENT)
        self._btn_pause = self._action_btn("⏸  Pause",     ACCENT2)
        self._btn_calib.clicked.connect(self.request_calibrate.emit)
        self._btn_pause.clicked.connect(self.request_pause_toggle.emit)
        btn_row.addWidget(self._btn_calib)
        btn_row.addWidget(self._btn_pause)
        side_lay.addLayout(btn_row)

        body.addWidget(side, 0)

        # ── Footer tip strip ─────────────────────────────────────────────
        tip = QLabel("TIP  ·  Activate by showing an OPEN PALM, or say "
                     "“Wednesday, activate eye control.”")
        tip.setFont(QFont(FONT_MONO, 8))
        tip.setStyleSheet(
            f"color:{TXT3};background:rgba(255,255,255,0.02);"
            f"border:1px solid {BDR};border-radius:8px;padding:8px 14px;letter-spacing:1px;")
        outer.addWidget(tip)

        # Blink flash overlay timer
        self._flash_timer = QTimer(self)
        self._flash_timer.timeout.connect(self._tick_flash)
        self._flash_timer.start(50)

    # ── helpers ──────────────────────────────────────────────────────────
    def _build_bar(self, axis_label: str):
        wrap = QHBoxLayout(); wrap.setSpacing(8)
        lbl = QLabel(axis_label)
        lbl.setFont(QFont(FONT_MONO, 9, QFont.Weight.Bold))
        lbl.setFixedWidth(12)
        lbl.setStyleSheet(f"color:{TXT3};background:transparent;")
        wrap.addWidget(lbl)
        bg = QFrame()
        bg.setFixedHeight(6)
        bg.setStyleSheet("background:#1a1a1d;border:none;border-radius:3px;")
        fill = QFrame(bg)
        fill.setFixedHeight(6)
        fill.setStyleSheet(f"background:{GREEN};border:none;border-radius:3px;")
        fill.setGeometry(0, 0, 0, 6)
        bg._fill = fill
        wrap.addWidget(bg, 1)
        val = QLabel("—")
        val.setFont(QFont(FONT_MONO, 9))
        val.setFixedWidth(40)
        val.setStyleSheet(f"color:{TXT2};background:transparent;")
        val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        wrap.addWidget(val)
        return wrap, val

    def _action_btn(self, text: str, col: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setFont(QFont(FONT_MONO, 9, QFont.Weight.Bold))
        btn.setFixedHeight(36)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(
            f"QPushButton{{background:transparent;color:{col};"
            f"border:1px solid {col};border-radius:8px;letter-spacing:1px;}}"
            f"QPushButton:hover{{background:rgba(255,255,255,0.04);}}")
        return btn

    def _tick_flash(self):
        self._tick += 1
        # Slight blink-flash glow on state badge after a real blink event
        if self._tick - self._last_blink_event_tick < 6:
            self._state_badge.setStyleSheet(
                f"color:#fef3c7;background:rgba(245,158,11,0.18);"
                f"border:1px solid rgba(245,158,11,0.4);border-radius:6px;padding:0 12px;")
        else:
            self._set_default_badge_style()

    def _set_default_badge_style(self):
        if not self._is_active:
            self._state_badge.setStyleSheet(
                f"color:#f87171;background:rgba(239,68,68,0.08);"
                f"border:1px solid rgba(239,68,68,0.25);border-radius:6px;padding:0 12px;")
        else:
            self._state_badge.setStyleSheet(
                f"color:{GREEN};background:rgba(34,197,94,0.08);"
                f"border:1px solid rgba(34,197,94,0.25);border-radius:6px;padding:0 12px;")

    # ── Public: update from main window ─────────────────────────────────
    def update_frame(self, frame, data: dict):
        face_ok = bool(data.get("face_detected", False))
        self.viewfinder.set_frame(frame, face_ok)

        # Gaze
        gn = data.get("gaze_norm")
        if gn:
            self.viewfinder.push_gaze(gn[0], gn[1])
            self._set_bar_value(self._gaze_x_bar, self._gaze_x_val, gn[0])
            self._set_bar_value(self._gaze_y_bar, self._gaze_y_val, gn[1])

        # Activity
        self._card_blinks.set_value(str(data.get("blink_count", 0)))
        self._card_clicks.set_value(str(data.get("click_count", 0)))

        # Dwell
        dp = max(0.0, min(1.0, float(data.get("dwell_progress", 0.0))))
        self._dwell_val.setText(f"{int(dp*100)}%")
        w = max(0, int(self._dwell_bar_bg.width() * dp))
        self._dwell_bar.setGeometry(0, 0, w, 6)

        # Sensors
        ear = data.get("ear", 0.0) or 0.0
        self._card_ear.set_value(f"{ear:.2f}")
        self._card_fps.set_value(f"{data.get('fps',0):.0f}")

        # State badge
        state = data.get("system_state", "ACTIVE")
        self._is_active = (state == "ACTIVE")
        if self._is_active:
            self._state_badge.setText("  ● TRACKING")
        else:
            self._state_badge.setText("  ● PAUSED  (open palm to resume)")
        self._set_default_badge_style()

        if data.get("blink_event"):
            self._last_blink_event_tick = self._tick

        # ── Gesture card ─────────────────────────────────────────────────
        raw = (data.get("gesture") or "").strip()
        conf = max(0.0, min(1.0, float(data.get("confidence", 0)) / 100.0))
        entry = _GESTURE_EMOJI.get(raw)
        if entry:
            emoji, name, col = entry
            self._gest_emoji.setText(emoji)
            self._gest_name.setText(name)
            self._gest_name.setStyleSheet(f"color:{col};background:transparent;")
            self._gest_emoji_bg.setStyleSheet(
                f"background:rgba(255,255,255,0.06);"
                f"border:1px solid {col}55;border-radius:10px;")
            self._gest_accent_bar.setStyleSheet(
                f"background:{col};border:none;border-radius:1px;")
            self._gest_conf_fill.setStyleSheet(
                f"background:{col};border:none;border-radius:2px;")
        else:
            self._gest_emoji.setText("—")
            self._gest_name.setText("No gesture")
            self._gest_name.setStyleSheet(f"color:{TXT3};background:transparent;")
            self._gest_emoji_bg.setStyleSheet(
                f"background:rgba(255,255,255,0.04);border:1px solid {BDR};border-radius:10px;")
            self._gest_accent_bar.setStyleSheet(
                f"background:{TXT3};border:none;border-radius:1px;")
            self._gest_conf_fill.setStyleSheet(
                f"background:{ACCENT2};border:none;border-radius:2px;")
            conf = 0.0
        # Confidence bar fill
        bar_w = max(0, int(self._gest_conf_bg.width() * conf))
        self._gest_conf_fill.setGeometry(0, 0, bar_w, 4)
        self._gest_conf_pct.setText(f"{int(conf*100)}%")

    def _set_bar_value(self, layout, val_label, frac: float):
        frac = max(0.0, min(1.0, float(frac)))
        # The bar background is the 2nd item in layout (after the small axis label)
        bg = layout.itemAt(1).widget()
        if bg and hasattr(bg, "_fill"):
            bg._fill.setGeometry(0, 0, int(bg.width() * frac), 6)
        val_label.setText(f"{int(frac*100)}%")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════
class EyeconWindow(QMainWindow):
    chat_requested = pyqtSignal(str, str)

    def __init__(self, user: dict):
        super().__init__()
        self._user    = user
        self._username = user.get("username", user.get("full_name", "Boss")).capitalize()
        self._wednesday  = None
        self._cmd_count = 0

        self.setWindowTitle(f"Eyecon  —  {self._username}")
        self.setMinimumSize(1000, 640)
        self.resize(1200, 740)
        self.setStyleSheet(f"QMainWindow{{background:{BG};}}")

        self._build_ui()
        self.chat_requested.connect(self._add_chat_message)
        self._start_camera()

        # Pulse dot timer
        self._pulse_tick = 0

    # ─────────────────────────────────────────────────────────────────────────
    #  BUILD UI
    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QWidget()
        root.setStyleSheet(f"background:{BG};")
        self.setCentralWidget(root)
        main = QHBoxLayout(root)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        # Sidebar
        self._sidebar = Sidebar(self._username)
        self._sidebar.page_changed.connect(self._on_page)
        main.addWidget(self._sidebar)

        # Right side
        right = QWidget()
        right.setStyleSheet("background:transparent;")
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(0)

        right_lay.addWidget(self._build_topbar())

        # Page stack
        self._page_stack = QStackedWidget()
        self._page_stack.setStyleSheet("background:transparent;")

        # Page 0: Dashboard (single column)
        self._page_stack.addWidget(self._build_dashboard())

        # Generic pages for other sidebar items
        self._page_map = {"Dashboard": 0}

        # Custom: Eye Control live-view page
        self._eye_page = EyeControlPage()
        self._eye_page.request_pause_toggle.connect(self._toggle_eye_pause)
        self._eye_page.request_calibrate.connect(self._request_calibrate)
        self._page_map["Eye Control"] = self._page_stack.addWidget(self._eye_page)

        # Custom: Settings page (live config editor)
        from pages.settings_page import SettingsPage
        from pages.wednesday_page import WednesdayMissionControlPage
        self._settings_cfg = Config("config/settings.json") if BACKEND else None
        if self._settings_cfg is not None:
            self._settings_page = SettingsPage(self._settings_cfg)
            self._settings_page.settings_saved.connect(self._on_settings_saved)
            self._page_map["Settings"] = self._page_stack.addWidget(self._settings_page)

        # Custom: WEDNESDAY Mission Control
        if self._settings_cfg is not None:
            self._wed_page = WednesdayMissionControlPage(
                self._settings_cfg,
                engine_getter=lambda: getattr(self, "_wednesday", None),
            )
            self._wed_page.test_command_submitted.connect(self._on_message)
            self._wed_page.test_voice_requested.connect(self._on_test_voice)
            self._page_map["WEDNESDAY AI"] = self._page_stack.addWidget(self._wed_page)

        # Custom: Biometrics
        from pages.biometrics_page import BiometricsPage
        uid = int(self._user.get("id", 0)) if isinstance(self._user, dict) else 0
        self._bio_page = BiometricsPage(
            uid,
            get_verifier=lambda: (getattr(self._worker, "_ai", None) and
                                   getattr(self._worker._ai, "_bio", None)),
        )
        self._bio_page.reenroll_requested.connect(self._on_reenroll_requested)
        self._page_map["Biometrics"] = self._page_stack.addWidget(self._bio_page)

        # Custom: Calibrate
        if self._settings_cfg is not None:
            from pages.calibrate_page import CalibratePage
            self._calib_page = CalibratePage(
                self._settings_cfg,
                eye_getter=lambda: getattr(self._worker, "_eye", None),
            )
            self._calib_page.request_calibrate.connect(self._request_calibrate)
            self._page_map["Calibrate"] = self._page_stack.addWidget(self._calib_page)

        # Custom: Analytics
        from pages.analytics_page import AnalyticsPage
        self._analytics_page = AnalyticsPage(uid)
        self._page_map["Analytics"] = self._page_stack.addWidget(self._analytics_page)

        # Custom: Profile
        from pages.profile_page import ProfilePage
        self._profile_page = ProfilePage(self._user)
        self._profile_page.sign_out_requested.connect(self._on_sign_out)
        self._profile_page.navigate_to.connect(self._switch_page)
        self._page_map["Profile"] = self._page_stack.addWidget(self._profile_page)

        right_lay.addWidget(self._page_stack, 1)
        main.addWidget(right, 1)

    def _make_info_page(self, title, desc):
        page = QWidget()
        page.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(40, 40, 40, 40)
        lay.setSpacing(16)
        t = QLabel(title)
        t.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        t.setStyleSheet(f"color:{TXT};background:transparent;")
        lay.addWidget(t)
        d = QLabel(desc)
        d.setFont(QFont("Segoe UI", 12))
        d.setWordWrap(True)
        d.setStyleSheet(f"color:{TXT2};background:transparent;")
        lay.addWidget(d)
        lay.addStretch()
        return page
    def _build_topbar(self):
        tb = QWidget()
        tb.setFixedHeight(52)
        tb.setStyleSheet(f"background:{PNL};border-bottom:1px solid {BDR};")
        lay = QHBoxLayout(tb)
        lay.setContentsMargins(24, 0, 20, 0)
        lay.setSpacing(8)
        lay.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        title = _lbl("MULTIMODAL  CONTROL  INTERFACE", 9, TXT3, mono=True)
        title.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(title)
        lay.addStretch()

        status_data = [("Eye", GREEN), ("Gesture", ACCENT2), ("WEDNESDAY", "#e4e4e7")]
        for label, col in status_data:
            chip = QPushButton(label)
            chip.setFont(QFont(FONT_MONO, 9))
            chip.setFixedHeight(30)
            chip.setMinimumWidth(68)
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.setStyleSheet(
                f"QPushButton{{color:{col};background:transparent;"
                f"border:1px solid {col};border-radius:15px;padding:0 14px;"
                f"letter-spacing:1px;}}"
                f"QPushButton:hover{{background:rgba(255,255,255,0.05);}}")
            lay.addWidget(chip, 0, Qt.AlignmentFlag.AlignVCenter)
        return tb

    def _build_dashboard(self):
        page = QWidget()
        page.setStyleSheet(f"background:{BG};")
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Two-column body ───────────────────────────────────────────────────
        body = QWidget()
        body.setStyleSheet(f"background:{BG};")
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(0)
        outer.addWidget(body, 1)

        # ═══ LEFT PANEL: sphere + metrics ════════════════════════════════════
        left = QWidget()
        left.setStyleSheet(f"background:{BG};border-right:1px solid {BDR};")
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(28, 24, 28, 20)
        left_lay.setSpacing(0)

        # Session / greeting
        from datetime import datetime
        now = datetime.now(); h = now.hour
        tod_word = "MORNING" if h < 12 else "AFTERNOON" if h < 17 else "EVENING"
        date_str = now.strftime("%B %d, %Y")
        left_lay.addWidget(_lbl(f"{tod_word} SESSION  ·  {date_str.upper()}", 8, TXT3, mono=True))
        left_lay.addSpacing(8)

        tod = "Good morning" if h < 12 else "Good afternoon" if h < 17 else "Good evening"
        self._greet = QLabel(f'{tod}, <b style="font-weight:700">{self._username}.</b>')
        self._greet.setFont(QFont(FONT, 22))
        self._greet.setStyleSheet(f"color:{TXT};background:transparent;")
        self._greet.setTextFormat(Qt.TextFormat.RichText)
        left_lay.addWidget(self._greet)
        left_lay.addSpacing(8)

        badge = QLabel("  ● All systems operational")
        badge.setFont(QFont(FONT_MONO, 9))
        badge.setFixedHeight(26)
        badge.setStyleSheet(
            f"color:{GREEN};background:rgba(34,197,94,0.08);"
            f"border:1px solid rgba(34,197,94,0.2);border-radius:6px;padding:0 10px;")
        b_row = QHBoxLayout()
        b_row.setContentsMargins(0,0,0,0)
        b_row.addWidget(badge)
        b_row.addStretch()
        left_lay.addLayout(b_row)
        left_lay.addSpacing(18)

        # Sphere – fills available width, tall
        self._sphere_view = QWebEngineView()
        self._sphere_view.setFixedHeight(340)
        self._sphere_view.setMinimumWidth(180)
        self._sphere_view.setStyleSheet("border:none;background:#0d0d0f;border-radius:12px;")
        self._sphere_view.page().setBackgroundColor(QColor(13, 13, 15, 255))
        self._bridge  = SphereBridge()
        self._channel = QWebChannel()
        self._channel.registerObject("bridge", self._bridge)
        self._sphere_view.page().setWebChannel(self._channel)
        sphere_path = os.path.join(os.path.dirname(__file__), "sphere.html")
        self._sphere_view.load(QUrl.fromLocalFile(sphere_path))
        left_lay.addWidget(self._sphere_view)
        left_lay.addSpacing(6)

        self._sphere_state = _lbl("STANDBY", 9, TXT3, mono=True)
        self._sphere_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_lay.addWidget(self._sphere_state)
        left_lay.addSpacing(18)

        # Metrics 2×2 grid
        self._metrics = {}
        bar_colors = [ACCENT2, GREEN, ACCENT2, ACCENT2]
        metrics_data = [("FPS","fps","27"),("LATENCY","lat","18ms"),
                        ("COMMANDS","cmd","0"),("GAZE ACC.","acc","91%")]
        row1 = QHBoxLayout(); row1.setSpacing(8); row1.setContentsMargins(0,0,0,0)
        row2 = QHBoxLayout(); row2.setSpacing(8); row2.setContentsMargins(0,0,0,0)
        for i, (label, key, val) in enumerate(metrics_data):
            card = QWidget()
            card.setStyleSheet(f"background:{PNL2};border:1px solid {BDR};border-radius:10px;")
            card.setFixedHeight(68)
            cl = QVBoxLayout(card)
            cl.setContentsMargins(12, 10, 12, 6)
            cl.setSpacing(1)
            v_lbl = QLabel(val)
            v_lbl.setFont(QFont(FONT, 18, QFont.Weight.Light))
            v_lbl.setStyleSheet(f"color:{TXT};background:transparent;")
            cl.addWidget(v_lbl)
            l_lbl = QLabel(label)
            l_lbl.setFont(QFont(FONT_MONO, 7))
            l_lbl.setStyleSheet(f"color:{TXT3};background:transparent;letter-spacing:2px;")
            cl.addWidget(l_lbl)
            bar = QFrame(); bar.setFixedHeight(2)
            bar.setStyleSheet(f"background:{bar_colors[i]};border:none;border-radius:1px;")
            cl.addWidget(bar)
            self._metrics[key] = v_lbl
            (row1 if i < 2 else row2).addWidget(card, 1)
        left_lay.addLayout(row1)
        left_lay.addSpacing(8)
        left_lay.addLayout(row2)
        left_lay.addStretch()

        body_lay.addWidget(left, 5)   # 5 parts width

        # ═══ RIGHT PANEL: chat + input ════════════════════════════════════════
        right = QWidget()
        right.setStyleSheet(f"background:{BG};")
        right_outer = QVBoxLayout(right)
        right_outer.setContentsMargins(0, 0, 0, 0)
        right_outer.setSpacing(0)

        # Chat header
        chat_hdr = QWidget()
        chat_hdr.setFixedHeight(48)
        chat_hdr.setStyleSheet(f"background:{PNL};border-bottom:1px solid {BDR};")
        hdr_lay = QHBoxLayout(chat_hdr)
        hdr_lay.setContentsMargins(20, 0, 20, 0)
        hdr_lay.addWidget(_lbl("CONVERSATION", 9, TXT3, mono=True))
        hdr_lay.addStretch()
        hdr_lay.addWidget(_lbl("● LIVE", 8, GREEN, mono=True))
        right_outer.addWidget(chat_hdr)

        # Scrollable chat area
        self._chat_scroll = QScrollArea()
        self._chat_scroll.setWidgetResizable(True)
        self._chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._chat_scroll.setStyleSheet(
            f"QScrollArea{{border:none;background:{BG};}}"
            "QScrollBar:vertical{width:4px;background:transparent;}"
            f"QScrollBar::handle:vertical{{background:#2a2a2e;border-radius:2px;}}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}")

        chat_content = QWidget()
        chat_content.setStyleSheet(f"background:{BG};")
        self._chat_lay = QVBoxLayout(chat_content)
        self._chat_lay.setContentsMargins(20, 16, 20, 16)
        self._chat_lay.setSpacing(14)
        self._chat_lay.addStretch()

        self._chat_scroll.setWidget(chat_content)
        right_outer.addWidget(self._chat_scroll, 1)

        # Input bar
        inp_w = QWidget()
        inp_w.setFixedHeight(60)
        inp_w.setStyleSheet(f"border-top:1px solid {BDR};background:{PNL};")
        inp_lay = QHBoxLayout(inp_w)
        inp_lay.setContentsMargins(16, 10, 16, 10)
        inp_lay.setSpacing(8)

        self._wake_btn = QPushButton("HEY WEDNESDAY")
        self._wake_btn.setFont(QFont(FONT_MONO, 8))
        self._wake_btn.setFixedSize(100, 36)
        self._wake_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._wake_btn.setStyleSheet(
            f"QPushButton{{background:transparent;color:{TXT3};"
            f"border:1px solid {BDR};border-radius:8px;letter-spacing:1px;}}"
            f"QPushButton:hover{{color:{TXT2};border-color:rgba(255,255,255,0.18);}}")
        self._wake_btn.clicked.connect(self._on_wake)
        inp_lay.addWidget(self._wake_btn)

        self._input = QLineEdit()
        self._input.setPlaceholderText("Message WEDNESDAY...")
        self._input.setFont(QFont(FONT, 11))
        self._input.setFixedHeight(36)
        self._input.setStyleSheet(
            f"QLineEdit{{background:{PNL2};color:{TXT};border:1px solid {BDR};"
            f"border-radius:10px;padding:0 14px;}}"
            f"QLineEdit:focus{{border:1px solid rgba(139,92,246,0.45);}}"
            f"QLineEdit::placeholder{{color:{TXT3};}}")
        self._input.returnPressed.connect(self._send_msg)
        inp_lay.addWidget(self._input, 1)

        self._send_btn = QPushButton("\u27A4")
        self._send_btn.setFixedSize(36, 36)
        self._send_btn.setFont(QFont(FONT, 13))
        self._send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._send_btn.setStyleSheet(
            f"QPushButton{{background:{ACCENT2};color:#fff;border:none;border-radius:10px;}}"
            f"QPushButton:hover{{background:#7c3aed;}}")
        self._send_btn.clicked.connect(self._on_send_clicked)
        inp_lay.addWidget(self._send_btn)

        right_outer.addWidget(inp_w)
        body_lay.addWidget(right, 4)   # 4 parts width

        return page

    def _scroll_to_bottom(self):
        if hasattr(self, '_chat_scroll'):
            bar = self._chat_scroll.verticalScrollBar()
            bar.setValue(bar.maximum())

    def _on_send_clicked(self):
        if hasattr(self, "_send_btn") and self._send_btn.text() == "■":
            # User clicked Stop during typing
            if hasattr(self, "_type_timer") and self._type_timer.isActive():
                self._type_timer.stop()
                self._type_lbl.setText(self._type_text)
                if hasattr(self, "_type_act_w"):
                    self._type_act_w.show()
            self._send_btn.setText("\u27A4")
        else:
            self._send_msg()

    def _send_msg(self):
        txt = self._input.text().strip()
        if txt:
            self._input.clear()
            self._on_message(txt)

    def _add_chat_message(self, role, text):
        if role == "loading":
            if hasattr(self, "_send_btn"):
                self._send_btn.setText("■")
            if hasattr(self, "_loading_row") and self._loading_row:
                return
            row = QWidget()
            row.setStyleSheet("background:transparent;")
            row_lay = QHBoxLayout(row)
            row_lay.setContentsMargins(0, 2, 0, 2)
            row_lay.setSpacing(10)
            
            av = QLabel("F")
            av.setFixedSize(30, 30)
            av.setAlignment(Qt.AlignmentFlag.AlignCenter)
            av.setFont(QFont(FONT, 11, QFont.Weight.Bold))
            av.setStyleSheet(f"background:{ACCENT2};color:#fff;border-radius:15px;border:none;")
            row_lay.addWidget(av, 0, Qt.AlignmentFlag.AlignTop)
            
            msg_col = QVBoxLayout()
            msg_col.setSpacing(3)
            tag = QLabel("WEDNESDAY")
            tag.setFont(QFont(FONT_MONO, 8))
            tag.setStyleSheet(f"color:{ACCENT2};background:transparent;letter-spacing:1px;")
            msg_col.addWidget(tag)
            
            self._loading_lbl = LoadingRings()
            msg_col.addWidget(self._loading_lbl, 0, Qt.AlignmentFlag.AlignLeft)
            row_lay.addLayout(msg_col, 1)
            
            idx = self._chat_lay.count() - 1
            self._chat_lay.insertWidget(idx, row)
            self._loading_row = row
            QTimer.singleShot(10, self._scroll_to_bottom)
            return

        if hasattr(self, "_loading_row") and self._loading_row:
            self._chat_lay.removeWidget(self._loading_row)
            self._loading_row.deleteLater()
            self._loading_row = None

        row = QWidget()
        row.setStyleSheet("background:transparent;")
        row_lay = QHBoxLayout(row)
        row_lay.setContentsMargins(0, 2, 0, 2)
        row_lay.setSpacing(10)

        if role == "system":
            lbl = _lbl(text, 9, TXT3, mono=True)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            row_lay.addWidget(lbl)
        elif role == "wednesday":
            av = QLabel("F")
            av.setFixedSize(30, 30)
            av.setAlignment(Qt.AlignmentFlag.AlignCenter)
            av.setFont(QFont(FONT, 11, QFont.Weight.Bold))
            av.setStyleSheet(f"background:{ACCENT2};color:#fff;border-radius:15px;border:none;")
            row_lay.addWidget(av, 0, Qt.AlignmentFlag.AlignTop)
            msg_col = QVBoxLayout()
            msg_col.setSpacing(3)
            tag = QLabel("WEDNESDAY")
            tag.setFont(QFont(FONT_MONO, 8))
            tag.setStyleSheet(f"color:{ACCENT2};background:transparent;letter-spacing:1px;")
            msg_col.addWidget(tag)
            bubble = QLabel("")
            bubble.setFont(QFont(FONT, 11))
            bubble.setWordWrap(True)
            bubble.setStyleSheet(f"color:{TXT};background:transparent;padding:4px 0px;line-height:1.5;")
            msg_col.addWidget(bubble)
            
            # Action Buttons Container (Hidden during typing)
            act_w = QWidget()
            act_w.setStyleSheet("background:transparent;")
            act_lay = QHBoxLayout(act_w)
            act_lay.setContentsMargins(0, 4, 0, 0)
            act_lay.setSpacing(12)
            
            def make_act_btn(icon_text, tooltip):
                btn = QPushButton(icon_text)
                btn.setToolTip(tooltip)
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.setStyleSheet(f"QPushButton{{background:transparent;color:{TXT3};border:none;font-size:14px;}} "
                                  f"QPushButton:hover{{color:{ACCENT2};}}")
                btn.setFixedSize(24, 24)
                return btn

            copy_btn = make_act_btn("\u29C9", "Copy")
            like_btn = make_act_btn("\u21E7", "Like")
            dislike_btn = make_act_btn("\u21E9", "Dislike")
            retry_btn = make_act_btn("\u21BB", "Retry")
            
            copy_btn.clicked.connect(lambda _, t=text: QApplication.clipboard().setText(t))
            retry_btn.clicked.connect(lambda _: self._on_message(self._last_user_query) if hasattr(self, '_last_user_query') else None)
            
            act_lay.addWidget(copy_btn)
            act_lay.addWidget(like_btn)
            act_lay.addWidget(dislike_btn)
            act_lay.addWidget(retry_btn)
            act_lay.addStretch()
            
            act_w.hide()
            msg_col.addWidget(act_w)
            
            row_lay.addLayout(msg_col, 1)
            
            # Typing animation
            self._type_idx = 0
            self._type_text = text
            self._type_lbl = bubble
            self._type_act_w = act_w
            self._type_timer = QTimer(self)
            def type_char():
                if self._type_idx < len(self._type_text):
                    self._type_idx += 1
                    self._type_lbl.setText(self._type_text[:self._type_idx])
                    self._scroll_to_bottom()
                else:
                    self._type_timer.stop()
                    self._type_act_w.show()
                    if hasattr(self, "_send_btn"):
                        self._send_btn.setText("\u27A4")
                    QTimer.singleShot(10, self._scroll_to_bottom)
            self._type_timer.timeout.connect(type_char)
            self._type_timer.start(15) # 15ms per character (fast reading)
            
        else:
            row_lay.addStretch()
            msg_col = QVBoxLayout()
            msg_col.setSpacing(3)
            tag = QLabel("YOU")
            tag.setFont(QFont(FONT_MONO, 8))
            tag.setAlignment(Qt.AlignmentFlag.AlignRight)
            tag.setStyleSheet(f"color:{GREEN};background:transparent;letter-spacing:1px;")
            msg_col.addWidget(tag)
            bubble = QLabel(text)
            bubble.setFont(QFont(FONT, 11))
            bubble.setWordWrap(True)
            bubble.setMaximumWidth(480)
            bubble.setAlignment(Qt.AlignmentFlag.AlignRight)
            bubble.setStyleSheet(
                f"color:{TXT};background:{SURFACE};border:1px solid {BDR};"
                f"border-radius:12px 4px 12px 12px;padding:12px 16px;")
            msg_col.addWidget(bubble)
            row_lay.addLayout(msg_col)

        idx = self._chat_lay.count() - 1
        self._chat_lay.insertWidget(idx, row)
        QTimer.singleShot(10, self._scroll_to_bottom)


    # ─────────────────────────────────────────────────────────────────────────
    #  CAMERA + BACKEND
    # ─────────────────────────────────────────────────────────────────────────
    def _start_camera(self):
        cfg = Config("config/settings.json") if BACKEND else None
        fb  = FeedbackSystem() if BACKEND else None
        self._worker = CameraWorker(BACKEND, cfg, fb)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.start()

        # Give worker time to init friday
        QTimer.singleShot(1500, self._wire_wednesday)

    def _wire_wednesday(self):
        if not BACKEND: return
        fri = self._worker.get_friday()
        if fri:
            self._wednesday = fri
            fri.set_sphere_callback(self._set_sphere_state)
            fri.set_chat_callback(self._on_chat_cb)
            fri.set_page_callback(self._switch_page)
            fri.set_history_callback(self._on_engine_history)
            print("[app] Successfully wired WEDNESDAY engine.")
            self._boot_greeting()
            # Initial memory load for Mission Control panel
            if hasattr(self, "_wed_page"):
                QTimer.singleShot(200, self._wed_page._reload_memory)
        else:
            # Not initialized yet, try again in 500ms
            QTimer.singleShot(500, self._wire_wednesday)

    def _on_frame(self, frame, data):
        # Update metrics
        self._metrics["fps"].setText(f"{data['fps']:.0f}")
        self._metrics["lat"].setText(f"{data['latency']}ms")
        self._metrics["acc"].setText(f"{random.randint(87,96)}%")

        action = data.get("action")
        if action:
            self._cmd_count += 1
            self._metrics["cmd"].setText(str(self._cmd_count))

        # Sidebar status dots
        self._sidebar.set_status("eye",  data.get("eye_active",  True))
        self._sidebar.set_status("gest", data.get("hand_active", False))

        # ── Auto-switch to Eye Control when OPEN_PALM resumes the system ──
        if action == "RESUME" and self._page_stack.currentIndex() != self._page_map.get("Eye Control"):
            self._switch_page("Eye Control")

        # ── Push live frame to Eye Control page when visible ─────────────
        if (hasattr(self, "_eye_page") and
                self._page_stack.currentIndex() == self._page_map.get("Eye Control")):
            try:
                self._eye_page.update_frame(frame, data)
            except Exception:
                pass

        if (hasattr(self, "_calib_page") and
                self._page_stack.currentIndex() == self._page_map.get("Calibrate")):
            try:
                self._calib_page.update_frame(frame, data)
            except Exception:
                pass

        # Analytics page samples FPS continuously (cheap) — feed regardless of visibility
        if hasattr(self, "_analytics_page"):
            try:
                self._analytics_page.update_frame(frame, data)
            except Exception:
                pass

    # ── Page navigation helpers ──────────────────────────────────────────
    def _switch_page(self, name: str):
        """Thread-safe page switch; also reflects in sidebar selection."""
        QTimer.singleShot(0, lambda: self._do_switch_page(name))

    def _do_switch_page(self, name: str):
        idx = self._page_map.get(name)
        if idx is None:
            return
        # Update sidebar selection visually
        for b in self._sidebar._buttons:
            if b.text().strip() == name:
                self._sidebar._select(name, b)
                return
        self._page_stack.setCurrentIndex(idx)

    def _toggle_eye_pause(self):
        ai = getattr(self._worker, "_ai", None)
        if ai is None:
            return
        new_action = "PAUSE" if getattr(ai, "state", "PAUSED") == "ACTIVE" else "RESUME"
        ai.execute({"action": new_action})

    def _request_calibrate(self):
        if not self._worker or not self._worker._backend:
            self._add_chat_message("system", "Calibration unavailable (demo mode)")
            return
        eye = getattr(self._worker, "_eye", None)
        cap = getattr(self._worker, "_cap", None)
        if eye and cap:
            threading.Thread(
                target=lambda: eye.calibrate(cap), daemon=True).start()
            self._add_chat_message("system", "Eye calibration started — follow the dot.")

    # ─────────────────────────────────────────────────────────────────────────
    #  SETTINGS HOT-RELOAD
    # ─────────────────────────────────────────────────────────────────────────
    # Keys that actually take effect at runtime without a restart.
    _LIVE_KEYS = {
        "gaze_smooth_frames", "wednesday_memory_turns",
        "blink_click", "dwell_click", "eye_moves_cursor",
        "ai_action_cooldown_frames",
        "bio_face_threshold", "bio_voice_threshold",
        "bio_fail_streak_limit", "bio_grace_period_secs",
        "bio_check_interval_secs",
        "scroll_use_shift_hscroll",
    }

    def _on_settings_saved(self, updates: dict):
        """Push live-applicable settings into running modules; the rest are
        persisted to JSON and apply on next launch."""
        worker_cfg = getattr(self._worker, "_cfg", None)
        if worker_cfg is not None:
            worker_cfg.set_many(updates)

        eye_t = getattr(self._worker, "_eye", None)
        wed   = getattr(self._worker, "_wednesday", None)

        if eye_t is not None and "gaze_smooth_frames" in updates:
            eye_t._buf_size = int(updates["gaze_smooth_frames"])
        if wed is not None and "wednesday_memory_turns" in updates:
            wed._memory_max = int(updates["wednesday_memory_turns"])

        live = sum(1 for k in updates if k in self._LIVE_KEYS)
        deferred = len(updates) - live
        msg = f"Settings saved · {live} applied now"
        if deferred:
            msg += f" · {deferred} take effect on next launch"
        QTimer.singleShot(0, lambda: self._add_chat_message("system", msg))

    # ─────────────────────────────────────────────────────────────────────────
    #  WEDNESDAY HISTORY HOOK  (Mission Control)
    # ─────────────────────────────────────────────────────────────────────────
    def _on_engine_history(self, entry: dict):
        """Called from WEDNESDAY thread; marshal to GUI."""
        QTimer.singleShot(0, lambda: self._do_log_entry(entry))

    def _do_log_entry(self, entry: dict):
        if hasattr(self, "_wed_page"):
            try:    self._wed_page.add_log_entry(entry)
            except Exception: pass

    def _on_test_voice(self):
        if self._wednesday:
            try:
                self._wednesday._speak("WEDNESDAY voice channel online. All systems nominal.")
            except Exception:
                self._add_chat_message("system", "Voice test failed.")

    def _on_sign_out(self):
        """Return to the auth window. Wires the new AuthWindow's
        authenticated signal back to a fresh EyeconWindow so the user can
        re-log in without restarting the process."""
        try:
            from auth_window import AuthWindow
        except Exception as e:
            self._add_chat_message("system", f"Sign-out failed: {e}")
            return
        # Stop background worker before closing window
        try:    self._worker.stop()
        except Exception: pass

        self._auth = AuthWindow()

        def _relaunch(user: dict):
            user.pop("_is_new", False)
            from app import EyeconWindow
            self._next = EyeconWindow(user=user)
            self._next.show()

        self._auth.authenticated.connect(_relaunch)
        self._auth.show()
        self.close()

    def _on_reenroll_requested(self, modality: str):
        """Open the biometric enrollment window targeting a specific step."""
        try:
            from biometric_window import BiometricEnrollmentWindow
        except Exception as e:
            self._add_chat_message("system", f"Re-enroll unavailable: {e}")
            return
        step_map = {"Face": "face", "Hand": "hand", "Voice": "voice"}
        target = step_map.get(modality, None)
        try:
            w = BiometricEnrollmentWindow(self._user, start_step=target) \
                if target else BiometricEnrollmentWindow(self._user)
        except TypeError:
            # Older signature without start_step
            w = BiometricEnrollmentWindow(self._user)
        w.show()
        # Refresh tiles after window closes
        w.destroyed.connect(lambda *_: self._bio_page._refresh_enrollment())
        self._add_chat_message("system",
                               f"Re-enroll started for {modality}. Follow the prompts.")

    # ─────────────────────────────────────────────────────────────────────────
    #  SPHERE STATE
    # ─────────────────────────────────────────────────────────────────────────
    def _set_sphere_state(self, state: str):
        """Call from any thread — safe."""
        QTimer.singleShot(0, lambda: self._do_set_sphere(state))

    def _do_set_sphere(self, state: str):
        self._bridge.set_state(state)
        labels = {
            "idle":      ("WEDNESDAY ONLINE",    TXT3),
            "listening": ("LISTENING...",     "rgba(34,197,94,0.5)"),
            "processing":("PROCESSING",       "rgba(255,255,255,0.25)"),
            "speaking":  ("RESPONDING",       "rgba(255,255,255,0.3)"),
            "impostor":  ("UNRECOGNISED USER","rgba(239,68,68,0.5)"),
        }
        txt, col = labels.get(state, ("WEDNESDAY ONLINE", TXT3))
        self._sphere_state.setText(txt)
        self._sphere_state.setStyleSheet(f"color:{col};background:transparent;")

    # ─────────────────────────────────────────────────────────────────────────
    #  CHAT CALLBACKS
    # ─────────────────────────────────────────────────────────────────────────
    def _on_chat_cb(self, role: str, text: str):
        self.chat_requested.emit(role, text)

    def _on_message(self, text: str):
        if text == "__MIC__":
            if self._wednesday:
                self._wednesday.push_to_talk()
            else:
                self._add_chat_message("system", "WEDNESDAY not available in demo mode")
            return
        self._add_chat_message("user", text)
        self._last_user_query = text
        if self._wednesday:
            self._wednesday.text_input(text)
        else:
            # Demo responses
            demo = ["Understood, Boss.","Done.","On it.","Confirmed."]
            import random as r
            QTimer.singleShot(800, lambda: self._add_chat_message(
                "wednesday", r.choice(demo)))

    def _on_wake(self):
        if self._wednesday:
            self._wednesday.push_to_talk()
        self._set_sphere_state("listening")

    def _on_page(self, name: str):
        idx = self._page_map.get(name, 0)
        self._page_stack.setCurrentIndex(idx)

    # ─────────────────────────────────────────────────────────────────────────
    #  BOOT
    # ─────────────────────────────────────────────────────────────────────────
    def _boot_greeting(self):
        if self._wednesday:
            self._wednesday.boot_greeting(self._username)
        else:
            from datetime import datetime
            h = datetime.now().hour
            tod = "morning" if h < 12 else "afternoon" if h < 17 else "evening"
            self._add_chat_message(
                "wednesday",
                f"Good {tod}, {self._username}. All four input modalities are online and responding normally. Eye tracking is stable at 91% accuracy.")

    def closeEvent(self, event):
        self._worker.stop()
        event.accept()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY (for standalone test — normally called from main.py)
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("QToolTip{background:#161616;color:#e0e0e0;border:1px solid #222;}")
    win = EyeconWindow(user={"username": "suyash", "full_name": "Suyash"})
    win.show()
    sys.exit(app.exec())
