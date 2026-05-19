"""
pages/_common.py
────────────────
Shared theme tokens and reusable widgets (Toggle, Slider, SectionCard).
Mirrors the constants in app.py so each page can be self-contained.
"""

from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QSlider, QLineEdit, QComboBox, QCheckBox, QSizePolicy,
)
from PyQt6.QtCore  import Qt, pyqtSignal, QPoint, QPropertyAnimation, QRect
from PyQt6.QtGui   import QFont, QColor, QPainter, QPen, QBrush


# ── Theme tokens (kept identical to app.py) ─────────────────────────────────
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
FONT      = "Segoe UI"
FONT_MONO = "Consolas"


def lbl(text, size=11, color=TXT, bold=False, mono=False):
    l = QLabel(text)
    f = QFont(FONT_MONO if mono else FONT, size,
              QFont.Weight.Bold if bold else QFont.Weight.Normal)
    if not mono:
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.3)
    l.setFont(f)
    l.setStyleSheet(f"color:{color};background:transparent;")
    return l


def section(text):
    """A faint mono section label."""
    l = QLabel(text)
    l.setFont(QFont(FONT_MONO, 8))
    l.setStyleSheet(f"color:{TXT3};background:transparent;letter-spacing:2px;")
    return l


# ═══════════════════════════════════════════════════════════════════════════
#  SectionCard — titled rounded container
# ═══════════════════════════════════════════════════════════════════════════
class SectionCard(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame{{background:{PNL2};border:1px solid {BDR};border-radius:12px;}}")
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(16, 14, 16, 14)
        self._lay.setSpacing(10)

        if title:
            head = QLabel(title)
            head.setFont(QFont(FONT_MONO, 9, QFont.Weight.Bold))
            head.setStyleSheet(
                f"color:{TXT};background:transparent;letter-spacing:2px;border:none;")
            self._lay.addWidget(head)

    def add(self, widget_or_layout):
        if isinstance(widget_or_layout, QWidget):
            self._lay.addWidget(widget_or_layout)
        else:
            self._lay.addLayout(widget_or_layout)


# ═══════════════════════════════════════════════════════════════════════════
#  Toggle — animated pill switch
# ═══════════════════════════════════════════════════════════════════════════
class Toggle(QCheckBox):
    """Compact pill switch.  Subclasses QCheckBox so it integrates with
    Qt's signal/state system seamlessly (use .isChecked() / toggled signal)."""

    def __init__(self, checked: bool = False, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(40, 22)
        self.setChecked(checked)
        self.stateChanged.connect(lambda *_: self.update())

    def hitButton(self, pos: QPoint) -> bool:
        return self.contentsRect().contains(pos)

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        on = self.isChecked()
        track = QColor(GREEN) if on else QColor("#3f3f46")
        p.setBrush(track); p.setPen(Qt.PenStyle.NoPen)
        r = self.rect().adjusted(0, 0, -1, -1)
        p.drawRoundedRect(r, 11, 11)

        # Knob
        knob_d = 16
        margin = 3
        x = r.right() - knob_d - margin if on else r.left() + margin
        y = r.top() + (r.height() - knob_d) // 2
        p.setBrush(QColor("#fafafa"))
        p.drawEllipse(x, y, knob_d, knob_d)
        p.end()


# ═══════════════════════════════════════════════════════════════════════════
#  LabeledSlider — horizontal slider with title, value readout, hint
# ═══════════════════════════════════════════════════════════════════════════
class LabeledSlider(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(self, title: str, vmin: float, vmax: float, value: float,
                 step: float = 1.0, hint: str = "", suffix: str = "",
                 decimals: int = 0, parent=None):
        super().__init__(parent)
        self._vmin = vmin; self._vmax = vmax; self._step = step
        self._suffix = suffix; self._decimals = decimals

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        head = QHBoxLayout(); head.setSpacing(8)
        ttl = QLabel(title)
        ttl.setFont(QFont(FONT_MONO, 8))
        ttl.setStyleSheet(
            f"color:{TXT2};background:transparent;letter-spacing:1px;border:none;")
        head.addWidget(ttl)
        head.addStretch()
        self._val_lbl = QLabel("")
        self._val_lbl.setFont(QFont(FONT_MONO, 9, QFont.Weight.Bold))
        self._val_lbl.setStyleSheet(
            f"color:{TXT};background:transparent;border:none;")
        head.addWidget(self._val_lbl)
        lay.addLayout(head)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        # Map float range onto integer slider
        self._scale = max(1, int(round(1 / step))) if step < 1 else 1
        self._slider.setMinimum(int(round(vmin * self._scale)))
        self._slider.setMaximum(int(round(vmax * self._scale)))
        self._slider.setSingleStep(max(1, int(round(step * self._scale))))
        self._slider.setValue(int(round(value * self._scale)))
        self._slider.setStyleSheet(_SLIDER_QSS)
        self._slider.valueChanged.connect(self._on_change)
        lay.addWidget(self._slider)

        if hint:
            h = QLabel(hint)
            h.setFont(QFont(FONT, 9))
            h.setStyleSheet(f"color:{TXT3};background:transparent;border:none;")
            h.setWordWrap(True)
            lay.addWidget(h)

        self._refresh_label(value)

    def _on_change(self, v_int: int):
        v = v_int / self._scale
        self._refresh_label(v)
        self.valueChanged.emit(v)

    def _refresh_label(self, v: float):
        if self._decimals > 0 or self._scale > 1:
            txt = f"{v:.{max(self._decimals, 0)}f}"
        else:
            txt = f"{int(v)}"
        self._val_lbl.setText(f"{txt}{self._suffix}")

    def value(self) -> float:
        return self._slider.value() / self._scale


_SLIDER_QSS = f"""
QSlider::groove:horizontal {{
    height: 4px; background: #1a1a1d; border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; background: #fafafa;
    margin: -6px 0; border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{
    background: #ffffff;
}}
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Row helper — left label + right control inside a SectionCard
# ═══════════════════════════════════════════════════════════════════════════
def control_row(label_text: str, control: QWidget) -> QHBoxLayout:
    row = QHBoxLayout(); row.setSpacing(10)
    l = QLabel(label_text)
    l.setFont(QFont(FONT, 10))
    l.setStyleSheet(f"color:{TXT};background:transparent;border:none;")
    row.addWidget(l, 1)
    row.addWidget(control, 0)
    return row


# ═══════════════════════════════════════════════════════════════════════════
#  Action button (matches Eye Control style)
# ═══════════════════════════════════════════════════════════════════════════
def action_button(text: str, color: str, fixed_h: int = 36) -> QPushButton:
    btn = QPushButton(text)
    btn.setFont(QFont(FONT_MONO, 9, QFont.Weight.Bold))
    btn.setFixedHeight(fixed_h)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setStyleSheet(
        f"QPushButton{{background:transparent;color:{color};"
        f"border:1px solid {color};border-radius:8px;letter-spacing:1px;"
        f"padding:0 14px;}}"
        f"QPushButton:hover{{background:rgba(255,255,255,0.05);}}")
    return btn


def primary_button(text: str, color: str = ACCENT, fixed_h: int = 36) -> QPushButton:
    btn = QPushButton(text)
    btn.setFont(QFont(FONT_MONO, 9, QFont.Weight.Bold))
    btn.setFixedHeight(fixed_h)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setStyleSheet(
        f"QPushButton{{background:{color};color:#0a0a0a;"
        f"border:none;border-radius:8px;letter-spacing:1px;padding:0 18px;}}"
        f"QPushButton:hover{{background:#34d399;}}")
    return btn


# ═══════════════════════════════════════════════════════════════════════════
#  Page header — title bar + status badge
# ═══════════════════════════════════════════════════════════════════════════
def page_header(icon: str, title: str, subtitle: str,
                badge_text: str = "", badge_color: str = GREEN):
    """Returns (header_widget, badge_label) — keep badge_label to update later."""
    w = QWidget(); w.setStyleSheet("background:transparent;")
    lay = QVBoxLayout(w); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(6)

    row = QHBoxLayout(); row.setSpacing(12)
    t = QLabel(f"{icon}  {title}")
    t.setFont(QFont(FONT_MONO, 11, QFont.Weight.Bold))
    t.setStyleSheet(f"color:{TXT};background:transparent;letter-spacing:3px;")
    row.addWidget(t)
    row.addStretch()
    badge = None
    if badge_text:
        badge = QLabel(f"  {badge_text}")
        badge.setFont(QFont(FONT_MONO, 9))
        badge.setFixedHeight(26)
        badge.setStyleSheet(
            f"color:{badge_color};"
            f"background:rgba(34,197,94,0.08);"
            f"border:1px solid rgba(34,197,94,0.25);"
            f"border-radius:6px;padding:0 12px;")
        row.addWidget(badge)
    lay.addLayout(row)

    s = QLabel(subtitle)
    s.setFont(QFont(FONT, 10))
    s.setStyleSheet(f"color:{TXT2};background:transparent;")
    lay.addWidget(s)
    return w, badge
