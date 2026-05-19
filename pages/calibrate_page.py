"""
pages/calibrate_page.py
───────────────────────
Calibration & threshold tuning.
 - Status card with the fitted calibration matrix readout
 - Live gaze visualization (normalized 0..1 dot on a viewport)
 - Threshold sliders that write to config (gaze_smooth_frames is live)
 - Run calibration button (delegates to existing OpenCV-window flow)
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout,
    QSizePolicy,
)
from PyQt6.QtCore  import Qt, pyqtSignal, QTimer, QRect
from PyQt6.QtGui   import QFont, QPainter, QPen, QBrush, QColor, QPainterPath

from ._common import (
    BG, PNL, PNL2, BDR, TXT, TXT2, TXT3, GREEN, ACCENT, ACCENT2,
    FONT, FONT_MONO, lbl, section, page_header,
    SectionCard, LabeledSlider, action_button, primary_button,
)


class _GazeViewport(QFrame):
    """Black canvas showing a 9-dot grid + a moving gaze dot in real time."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(280)
        self.setStyleSheet(
            f"QFrame{{background:#000;border:1px solid {BDR};border-radius:14px;}}")
        self._gaze = None     # (nx, ny) in 0..1
        self._has_face = False

    def push_gaze(self, gaze_norm, face: bool):
        self._gaze = gaze_norm
        self._has_face = face
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(float(rect.x()), float(rect.y()),
                            float(rect.width()), float(rect.height()), 14, 14)
        p.setClipPath(path)
        p.fillRect(rect, QColor("#0a0a0c"))

        # 9-point grid (subtle)
        p.setPen(QPen(QColor(255, 255, 255, 36), 1))
        for fy in (0.15, 0.5, 0.85):
            for fx in (0.15, 0.5, 0.85):
                cx = int(rect.x() + fx * rect.width())
                cy = int(rect.y() + fy * rect.height())
                p.setBrush(QColor(255, 255, 255, 14))
                p.drawEllipse(cx - 9, cy - 9, 18, 18)
                p.setBrush(QColor(120, 120, 130))
                p.drawEllipse(cx - 3, cy - 3, 6, 6)

        # Centre crosshair
        cx = rect.center().x(); cy = rect.center().y()
        p.setPen(QPen(QColor(255, 255, 255, 22), 1))
        p.drawLine(cx - 14, cy, cx + 14, cy)
        p.drawLine(cx, cy - 14, cx, cy + 14)

        # Gaze dot
        if self._gaze and self._has_face:
            nx, ny = self._gaze
            nx = max(0.0, min(1.0, float(nx)))
            ny = max(0.0, min(1.0, float(ny)))
            gx = int(rect.x() + nx * rect.width())
            gy = int(rect.y() + ny * rect.height())
            # Glow
            for r, a in ((22, 30), (15, 60), (8, 120), (4, 220)):
                col = QColor(ACCENT); col.setAlpha(a)
                p.setBrush(col); p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(gx - r, gy - r, r*2, r*2)

        # Hint label
        if not self._has_face:
            p.setPen(QColor(TXT3))
            p.setFont(QFont(FONT_MONO, 9))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                       "no face detected — look at the camera")
        p.end()


class CalibratePage(QWidget):
    request_calibrate = pyqtSignal()

    # Keys that take effect live without restart
    _LIVE_KEYS = {"gaze_smooth_frames"}

    def __init__(self, config, eye_getter, parent=None):
        """eye_getter — callable returning live EyeTracker or None."""
        super().__init__(parent)
        self._cfg = config
        self._get_eye = eye_getter
        self._dirty: dict = {}

        self.setStyleSheet(f"background:{BG};")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 16, 28, 12)
        outer.setSpacing(10)

        head, self._badge = page_header(
            "⌖",
            "C  A  L  I  B  R  A  T  E",
            "Tune gaze accuracy and click thresholds. Run the 9-point "
            "calibration to fit a fresh mapping.",
            badge_text="● UNCALIBRATED",
            badge_color="#fbbf24",
        )
        outer.addWidget(head)

        body = QHBoxLayout(); body.setSpacing(14)
        outer.addLayout(body, 1)

        # ── Left: viewport + status ────────────────────────────────────
        left = QWidget(); left.setStyleSheet("background:transparent;")
        ll = QVBoxLayout(left); ll.setContentsMargins(0,0,0,0); ll.setSpacing(10)

        ll.addWidget(section("LIVE  ·  GAZE VIEWPORT"))
        self._viewport = _GazeViewport()
        ll.addWidget(self._viewport, 1)

        # Status card
        s = SectionCard("CALIBRATION STATE")
        self._status_lbl = QLabel("Not calibrated · using identity mapping")
        self._status_lbl.setFont(QFont(FONT_MONO, 9, QFont.Weight.Bold))
        self._status_lbl.setStyleSheet(f"color:#fbbf24;background:transparent;border:none;")
        s.add(self._status_lbl)

        self._matrix_lbl = QLabel("matrix: —")
        self._matrix_lbl.setFont(QFont(FONT_MONO, 8))
        self._matrix_lbl.setWordWrap(True)
        self._matrix_lbl.setStyleSheet(f"color:{TXT3};background:transparent;border:none;")
        s.add(self._matrix_lbl)

        # Action row
        actions = QHBoxLayout(); actions.setSpacing(8)
        self._run_btn = primary_button("▶  Run 9-point calibration", ACCENT)
        self._run_btn.clicked.connect(self.request_calibrate.emit)
        actions.addWidget(self._run_btn)
        actions.addStretch()
        s.add(actions)

        ll.addWidget(s)
        body.addWidget(left, 6)

        # ── Right: thresholds ──────────────────────────────────────────
        right = QWidget(); right.setStyleSheet("background:transparent;")
        rl = QVBoxLayout(right); rl.setContentsMargins(0,0,0,0); rl.setSpacing(10)

        c = SectionCard("THRESHOLDS")
        for key, title, vmin, vmax, step, dec, sfx, hint in [
            ("gaze_smooth_frames", "Gaze smoothing frames",
             3, 12, 1, 0, "", "Live · higher = smoother but laggier"),
            ("eye_inactive_secs",  "Inactivity timeout",
             5, 60, 1, 0, " s", "Eye control disengages after this idle period"),
            ("pinch_distance_thresh", "Pinch threshold",
             0.03, 0.10, 0.005, 3, "", "Smaller = more sensitive pinch"),
            ("gesture_global_cooldown", "Gesture cooldown",
             4, 40, 1, 0, " f", "Frames between accepted gestures"),
            ("ai_action_cooldown_frames", "AI action cooldown",
             0, 60, 1, 0, " f", "Frames between AI dispatched actions"),
        ]:
            v = float(self._cfg.get(key, vmin))
            slider = LabeledSlider(title, vmin, vmax, v,
                                   step=step, decimals=dec, suffix=sfx, hint=hint)
            slider.valueChanged.connect(lambda val, k=key: self._mark_dirty(k, val))
            c.add(slider)
        rl.addWidget(c)

        # Reset / Save row
        bar = QFrame()
        bar.setStyleSheet(
            f"QFrame{{background:{PNL2};border:1px solid {BDR};border-radius:10px;}}")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(14, 8, 14, 8); bl.setSpacing(10)
        self._dirty_lbl = QLabel("0 unsaved changes")
        self._dirty_lbl.setFont(QFont(FONT_MONO, 9))
        self._dirty_lbl.setStyleSheet(f"color:{TXT3};background:transparent;border:none;")
        bl.addWidget(self._dirty_lbl)
        bl.addStretch()
        save = primary_button("💾  Save", ACCENT)
        save.clicked.connect(self._save)
        bl.addWidget(save)
        rl.addWidget(bar)
        rl.addStretch()

        body.addWidget(right, 5)

        # Periodic refresh of state from EyeTracker
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(150)

    # ─────────────────────────────────────────────────────────────────────
    def _mark_dirty(self, key: str, value):
        if self._cfg.get(key) == value:
            self._dirty.pop(key, None)
        else:
            self._dirty[key] = value
        n = len(self._dirty)
        self._dirty_lbl.setText(f"{n} unsaved change{'s' if n != 1 else ''}")
        self._dirty_lbl.setStyleSheet(
            f"color:{('#fbbf24' if n else TXT3)};background:transparent;border:none;")

    def _save(self):
        if not self._dirty: return
        coerced = {}
        for k, v in self._dirty.items():
            cur = self._cfg.get(k)
            if isinstance(cur, bool):    coerced[k] = bool(v)
            elif isinstance(cur, int):   coerced[k] = int(v)
            elif isinstance(cur, float): coerced[k] = float(v)
            else:                        coerced[k] = v
        self._cfg.set_many(coerced)
        try:
            self._cfg.save()
        except Exception as e:
            self._dirty_lbl.setText(f"Save failed: {e}")
            return
        # Apply live keys directly
        eye = self._get_eye()
        if eye is not None and "gaze_smooth_frames" in coerced:
            eye._buf_size = int(coerced["gaze_smooth_frames"])
        self._dirty.clear()
        self._dirty_lbl.setText("saved · 1 of changes applied live")

    # ─────────────────────────────────────────────────────────────────────
    def _tick(self):
        eye = self._get_eye()
        if eye is None:
            return
        # Calibration state
        cm = getattr(eye, "cal_matrix", None)
        if cm is not None and self._badge:
            self._badge.setText("  ● CALIBRATED")
            self._badge.setStyleSheet(
                f"color:{GREEN};background:rgba(34,197,94,0.08);"
                f"border:1px solid rgba(34,197,94,0.25);"
                f"border-radius:6px;padding:0 12px;")
            self._status_lbl.setText("Calibrated · 9-point linear mapping fitted")
            self._status_lbl.setStyleSheet(f"color:{GREEN};background:transparent;border:none;")
            try:
                arr = np.asarray(cm)
                rows = ["[ " + "  ".join(f"{x:+7.3f}" for x in row) + " ]"
                        for row in arr]
                self._matrix_lbl.setText("matrix:\n" + "\n".join(rows))
            except Exception:
                pass
        elif self._badge:
            self._badge.setText("  ● UNCALIBRATED")
            self._badge.setStyleSheet(
                f"color:#fbbf24;background:rgba(251,191,36,0.08);"
                f"border:1px solid rgba(251,191,36,0.3);"
                f"border-radius:6px;padding:0 12px;")

    def update_frame(self, frame, data: dict):
        gn = data.get("gaze_norm")
        face = bool(data.get("face_detected", False))
        self._viewport.push_gaze(gn, face)
