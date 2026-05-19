"""
pages/analytics_page.py
───────────────────────
Usage analytics — KPI cards, gesture frequency bar chart, FPS line chart,
and a session history table.  All charts drawn with QPainter (no
matplotlib runtime dependency).
"""

import os
import csv
import collections
from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout,
    QScrollArea, QSizePolicy,
)
from PyQt6.QtCore  import Qt, QTimer, QRect
from PyQt6.QtGui   import QFont, QPainter, QPen, QBrush, QColor, QPainterPath

from ._common import (
    BG, PNL, PNL2, BDR, TXT, TXT2, TXT3, GREEN, ACCENT, ACCENT2,
    FONT, FONT_MONO, lbl, section, page_header,
    SectionCard,
)


# ═══════════════════════════════════════════════════════════════════════════
class _KPICard(QFrame):
    def __init__(self, title: str, accent: str = ACCENT):
        super().__init__()
        self.setStyleSheet(
            f"QFrame{{background:{PNL2};border:1px solid {BDR};border-radius:12px;}}")
        self.setFixedHeight(96)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12); lay.setSpacing(2)
        t = QLabel(title.upper())
        t.setFont(QFont(FONT_MONO, 8))
        t.setStyleSheet(
            f"color:{TXT3};background:transparent;border:none;letter-spacing:2px;")
        lay.addWidget(t)
        self._val = QLabel("0")
        self._val.setFont(QFont(FONT, 26, QFont.Weight.Bold))
        self._val.setStyleSheet(f"color:{TXT};background:transparent;border:none;")
        lay.addWidget(self._val)
        self._sub = QLabel("")
        self._sub.setFont(QFont(FONT_MONO, 8))
        self._sub.setStyleSheet(f"color:{accent};background:transparent;border:none;")
        lay.addWidget(self._sub)

    def set(self, value: str, sub: str = ""):
        self._val.setText(value)
        self._sub.setText(sub)


# ═══════════════════════════════════════════════════════════════════════════
class _BarChart(QFrame):
    """Horizontal-bar chart for gesture frequencies."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame{{background:{PNL2};border:1px solid {BDR};border-radius:12px;}}")
        self.setMinimumHeight(220)
        self._data = []   # list of (label, value, color)

    def set_data(self, items):
        self._data = list(items)
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(16, 16, -16, -16)
        if not self._data:
            p.setPen(QColor(TXT3))
            p.setFont(QFont(FONT_MONO, 9))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                       "no gesture activity yet")
            p.end(); return

        max_v = max(v for _, v, _ in self._data) or 1
        n = len(self._data)
        row_h = max(18, rect.height() // max(n, 1))
        for i, (label, v, col) in enumerate(self._data):
            y = rect.y() + i * row_h
            label_w = 110
            # Label
            p.setPen(QColor(TXT2))
            p.setFont(QFont(FONT_MONO, 9))
            p.drawText(rect.x(), y, label_w, row_h,
                       Qt.AlignmentFlag.AlignVCenter, label)
            # Bar bg
            bar_x = rect.x() + label_w
            bar_w_total = rect.width() - label_w - 50
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 14))
            p.drawRoundedRect(bar_x, y + row_h//2 - 5, bar_w_total, 10, 5, 5)
            # Bar fill
            w = int(bar_w_total * (v / max_v))
            p.setBrush(QColor(col))
            p.drawRoundedRect(bar_x, y + row_h//2 - 5, w, 10, 5, 5)
            # Value
            p.setPen(QColor(TXT))
            p.setFont(QFont(FONT_MONO, 9, QFont.Weight.Bold))
            p.drawText(bar_x + bar_w_total + 6, y, 50, row_h,
                       Qt.AlignmentFlag.AlignVCenter, str(v))
        p.end()


# ═══════════════════════════════════════════════════════════════════════════
class _LineChart(QFrame):
    """Rolling FPS line chart."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame{{background:{PNL2};border:1px solid {BDR};border-radius:12px;}}")
        self.setMinimumHeight(180)
        self._samples = collections.deque(maxlen=120)

    def push(self, v: float):
        self._samples.append(float(v))
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(20, 20, -20, -20)

        # Gridlines
        p.setPen(QPen(QColor(255, 255, 255, 14), 1))
        for f in (0.25, 0.5, 0.75):
            y = int(rect.y() + f * rect.height())
            p.drawLine(rect.x(), y, rect.right(), y)

        # Labels
        p.setPen(QColor(TXT3))
        p.setFont(QFont(FONT_MONO, 7))
        for f, lbl in ((0.0, "60"), (0.5, "30"), (1.0, "0")):
            y = int(rect.y() + f * rect.height())
            p.drawText(rect.x() - 18, y - 6, 16, 12,
                       Qt.AlignmentFlag.AlignRight, lbl)

        if len(self._samples) < 2:
            p.setPen(QColor(TXT3))
            p.setFont(QFont(FONT_MONO, 9))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                       "collecting fps samples…")
            p.end(); return

        n = len(self._samples)
        dx = rect.width() / (n - 1)
        pts = []
        for i, v in enumerate(self._samples):
            v_clamped = max(0.0, min(60.0, v)) / 60.0
            x = rect.x() + i * dx
            y = rect.bottom() - v_clamped * rect.height()
            pts.append((x, y))
        # Fill
        col_fill = QColor(ACCENT2); col_fill.setAlpha(60)
        path = QPainterPath()
        path.moveTo(pts[0][0], rect.bottom())
        for x, y in pts: path.lineTo(x, y)
        path.lineTo(pts[-1][0], rect.bottom())
        path.closeSubpath()
        p.fillPath(path, QBrush(col_fill))
        # Line
        p.setPen(QPen(QColor(ACCENT2), 2))
        for i in range(1, n):
            p.drawLine(int(pts[i-1][0]), int(pts[i-1][1]),
                       int(pts[i][0]),   int(pts[i][1]))
        # Last value label
        lv = self._samples[-1]
        p.setPen(QColor(TXT))
        p.setFont(QFont(FONT_MONO, 10, QFont.Weight.Bold))
        p.drawText(int(pts[-1][0]) - 30, int(pts[-1][1]) - 14, 60, 14,
                   Qt.AlignmentFlag.AlignCenter, f"{lv:.0f} fps")
        p.end()


# ═══════════════════════════════════════════════════════════════════════════
class AnalyticsPage(QWidget):
    """Analytics dashboard."""

    _PALETTE = ["#22c55e", "#8b5cf6", "#38bdf8", "#f59e0b",
                "#ef4444", "#34d399", "#fb923c", "#a78bfa", "#818cf8"]

    def __init__(self, user_id: int, parent=None):
        super().__init__(parent)
        self._user_id = user_id
        self._gesture_counts = collections.Counter()  # this-session counts
        self.setStyleSheet(f"background:{BG};")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 16, 28, 12)
        outer.setSpacing(10)

        head, self._badge = page_header(
            "📊",
            "A  N  A  L  Y  T  I  C  S",
            "Session metrics, gesture frequency and FPS stability.",
            badge_text="● LIVE",
        )
        outer.addWidget(head)

        # KPI row
        kpis = QHBoxLayout(); kpis.setSpacing(12)
        self._k_sessions = _KPICard("Sessions",       ACCENT)
        self._k_commands = _KPICard("Total commands", ACCENT2)
        self._k_blinks   = _KPICard("Total blinks",   "#38bdf8")
        self._k_avg_dur  = _KPICard("Avg duration",   "#f59e0b")
        for k in (self._k_sessions, self._k_commands, self._k_blinks, self._k_avg_dur):
            kpis.addWidget(k, 1)
        outer.addLayout(kpis)

        # Charts row
        charts = QHBoxLayout(); charts.setSpacing(12)
        bar_card = SectionCard("GESTURE FREQUENCY  ·  this session")
        bar_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._bar = _BarChart()
        bar_card.add(self._bar)
        charts.addWidget(bar_card, 1)

        line_card = SectionCard("FPS  ·  rolling 120 frames")
        line_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._line = _LineChart()
        line_card.add(self._line)
        charts.addWidget(line_card, 1)
        outer.addLayout(charts, 1)

        # Session history
        hist = SectionCard("SESSION HISTORY  ·  recent")
        hist.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._hist_inner = QWidget(); self._hist_inner.setStyleSheet("background:transparent;")
        self._hist_lay = QVBoxLayout(self._hist_inner)
        self._hist_lay.setContentsMargins(0, 0, 0, 0); self._hist_lay.setSpacing(2)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(160)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea{{border:1px solid {BDR};border-radius:8px;background:#0f0f12;}}"
            "QScrollBar:vertical{width:6px;background:transparent;}"
            f"QScrollBar::handle:vertical{{background:#2a2a2e;border-radius:3px;}}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}")
        scroll.setWidget(self._hist_inner)
        hist.add(scroll)
        outer.addWidget(hist)

        # Periodic refresh of CSV-backed parts
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_csv)
        self._timer.start(5000)
        self._refresh_csv()

    # ─────────────────────────────────────────────────────────────────────
    def update_frame(self, frame, data: dict):
        # Push fps
        try:
            self._line.push(float(data.get("fps", 0)))
        except Exception:
            pass
        # Tally executed gestures
        if data.get("gesture_executed"):
            g = data.get("gesture") or "—"
            if g and g != "—":
                self._gesture_counts[g] += 1
                self._refresh_bar()

    def _refresh_bar(self):
        items = []
        for i, (g, c) in enumerate(self._gesture_counts.most_common(8)):
            items.append((g, c, self._PALETTE[i % len(self._PALETTE)]))
        self._bar.set_data(items)

    # ─────────────────────────────────────────────────────────────────────
    def _refresh_csv(self):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "sessions.csv")
        path = os.path.abspath(path)
        rows = []
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    rows = [r for r in csv.DictReader(f)
                            if str(r.get("user_id", "")) == str(self._user_id)
                            or self._user_id == 0]
            except Exception:
                rows = []

        # KPIs
        n_sessions = len(rows)
        total_cmds = sum(int(r.get("total_commands", 0) or 0) for r in rows)
        total_blinks = sum(int(r.get("blink_clicks", 0) or 0) for r in rows)
        durs = [int(r.get("duration_secs", 0) or 0) for r in rows]
        avg_dur = sum(durs) / len(durs) if durs else 0
        self._k_sessions.set(str(n_sessions),
                             "—" if not rows else "logged sessions")
        self._k_commands.set(str(total_cmds),
                             "—" if not total_cmds else "across all sessions")
        self._k_blinks.set(str(total_blinks),
                           "—" if not total_blinks else "blink-driven clicks")
        if avg_dur > 60:
            avg_str = f"{avg_dur/60:.1f}m"
        else:
            avg_str = f"{int(avg_dur)}s"
        self._k_avg_dur.set(avg_str, "—" if not durs else "per session")

        # History rows
        while self._hist_lay.count():
            it = self._hist_lay.takeAt(0)
            if it.widget(): it.widget().deleteLater()
        if not rows:
            empty = QLabel("· no sessions yet — log out and back in to record one ·")
            empty.setFont(QFont(FONT_MONO, 9))
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setStyleSheet(f"color:{TXT3};background:transparent;border:none;padding:24px;")
            self._hist_lay.addWidget(empty)
        else:
            for r in reversed(rows[-20:]):
                self._hist_lay.addWidget(self._build_hist_row(r))
        self._hist_lay.addStretch()

    def _build_hist_row(self, r: dict) -> QWidget:
        w = QFrame()
        w.setStyleSheet(
            f"QFrame{{background:transparent;border:none;border-bottom:1px solid {BDR};border-radius:0;}}")
        ly = QHBoxLayout(w)
        ly.setContentsMargins(10, 5, 10, 5); ly.setSpacing(10)

        def cell(text, w_px=None, color=TXT2, mono=True, bold=False):
            l = QLabel(text)
            l.setFont(QFont(FONT_MONO if mono else FONT, 9,
                            QFont.Weight.Bold if bold else QFont.Weight.Normal))
            l.setStyleSheet(f"color:{color};background:transparent;border:none;")
            if w_px: l.setFixedWidth(w_px)
            return l

        ly.addWidget(cell(r.get("started_at", "")[:16], 130, TXT3))
        dur = int(r.get("duration_secs", 0) or 0)
        if dur > 60: ds = f"{dur/60:.1f}m"
        else:        ds = f"{dur}s"
        ly.addWidget(cell(ds, 60, TXT, bold=True))
        ly.addWidget(cell(f"{r.get('total_commands', 0)} cmd", 70, ACCENT))
        ly.addWidget(cell(f"{r.get('gestures_fired', 0)} gest", 70, ACCENT2))
        ly.addWidget(cell(f"{r.get('voice_cmds_fired', 0)} voice", 80, "#38bdf8"))
        ly.addWidget(cell(f"{r.get('avg_fps', '—')} fps avg", 90, TXT3))
        ly.addStretch()
        return w
