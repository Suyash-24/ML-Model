"""
pages/biometrics_page.py
────────────────────────
Biometric identity & security dashboard.
3 enrollment tiles · live face score · security log.
"""

import os
import csv
from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout,
    QScrollArea, QPushButton, QSizePolicy,
)
from PyQt6.QtCore  import Qt, pyqtSignal, QTimer
from PyQt6.QtGui   import QFont

from ._common import (
    BG, PNL, PNL2, BDR, TXT, TXT2, TXT3, GREEN, ACCENT, ACCENT2,
    FONT, FONT_MONO, lbl, section, page_header,
    SectionCard, action_button,
)


class _EnrollTile(QFrame):
    reenroll_clicked = pyqtSignal(str)   # emits modality name

    def __init__(self, modality: str, icon: str, parent=None):
        super().__init__(parent)
        self._modality = modality
        self._enrolled = False

        self.setStyleSheet(
            f"QFrame{{background:{PNL2};border:1px solid {BDR};border-radius:12px;}}")
        self.setFixedHeight(150)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14); lay.setSpacing(6)

        head = QHBoxLayout(); head.setSpacing(8)
        icon_l = QLabel(icon)
        icon_l.setFont(QFont("Segoe UI Emoji", 20))
        icon_l.setStyleSheet("background:transparent;border:none;")
        head.addWidget(icon_l)
        name_l = QLabel(modality.upper())
        name_l.setFont(QFont(FONT_MONO, 10, QFont.Weight.Bold))
        name_l.setStyleSheet(f"color:{TXT};background:transparent;border:none;letter-spacing:2px;")
        head.addWidget(name_l)
        head.addStretch()
        self._status_pill = QLabel("· not enrolled ·")
        self._status_pill.setFont(QFont(FONT_MONO, 8, QFont.Weight.Bold))
        self._status_pill.setFixedHeight(22)
        self._set_pill(False)
        head.addWidget(self._status_pill)
        lay.addLayout(head)

        self._meta = QLabel("Enroll to enable identity verification.")
        self._meta.setFont(QFont(FONT, 9))
        self._meta.setWordWrap(True)
        self._meta.setStyleSheet(f"color:{TXT3};background:transparent;border:none;")
        lay.addWidget(self._meta)

        lay.addStretch()

        btn_row = QHBoxLayout(); btn_row.setSpacing(8)
        btn_row.addStretch()
        self._btn = action_button("⟲  Re-enroll", ACCENT2, fixed_h=30)
        self._btn.clicked.connect(lambda: self.reenroll_clicked.emit(self._modality))
        btn_row.addWidget(self._btn)
        lay.addLayout(btn_row)

    def _set_pill(self, enrolled: bool):
        if enrolled:
            self._status_pill.setText("  ● ENROLLED  ")
            self._status_pill.setStyleSheet(
                f"color:{GREEN};background:rgba(34,197,94,0.1);"
                f"border:1px solid rgba(34,197,94,0.3);border-radius:11px;padding:0 8px;")
        else:
            self._status_pill.setText("  ○ NOT ENROLLED  ")
            self._status_pill.setStyleSheet(
                f"color:{TXT3};background:rgba(255,255,255,0.04);"
                f"border:1px solid {BDR};border-radius:11px;padding:0 8px;")

    def set_state(self, enrolled: bool, meta: str):
        self._enrolled = enrolled
        self._set_pill(enrolled)
        self._meta.setText(meta)


class BiometricsPage(QWidget):
    reenroll_requested = pyqtSignal(str)

    def __init__(self, user_id: int, get_verifier, parent=None):
        """get_verifier: callable returning live BiometricVerifier or None."""
        super().__init__(parent)
        self._user_id = user_id
        self._get_verifier = get_verifier
        self.setStyleSheet(f"background:{BG};")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 16, 28, 12)
        outer.setSpacing(10)

        head, self._badge = page_header(
            "🔒",
            "B  I  O  M  E  T  R  I  C  S",
            "Identity verification status, live similarity score and security log.",
            badge_text="● SECURED",
        )
        outer.addWidget(head)

        # Enrollment tiles row
        tiles = QHBoxLayout(); tiles.setSpacing(12)
        self._tile_face = _EnrollTile("Face",  "👤")
        self._tile_hand = _EnrollTile("Hand",  "✋")
        self._tile_voice= _EnrollTile("Voice", "🎙")
        for t in (self._tile_face, self._tile_hand, self._tile_voice):
            t.reenroll_clicked.connect(self.reenroll_requested.emit)
            tiles.addWidget(t, 1)
        outer.addLayout(tiles)

        # Live verification + Security log row
        body = QHBoxLayout(); body.setSpacing(12)
        outer.addLayout(body, 1)

        # Live face score card
        live = SectionCard("LIVE  ·  FACE MATCH SCORE")
        live.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._score_value = QLabel("—")
        self._score_value.setFont(QFont(FONT, 44, QFont.Weight.Light))
        self._score_value.setStyleSheet(f"color:{TXT};background:transparent;border:none;")
        live.add(self._score_value)

        # Threshold bar
        self._bar_bg = QFrame()
        self._bar_bg.setFixedHeight(8)
        self._bar_bg.setStyleSheet("background:#1a1a1d;border:none;border-radius:4px;")
        self._bar_fill = QFrame(self._bar_bg)
        self._bar_fill.setFixedHeight(8)
        self._bar_fill.setStyleSheet(f"background:{ACCENT};border:none;border-radius:4px;")
        self._bar_fill.setGeometry(0, 0, 0, 8)
        self._thresh_marker = QFrame(self._bar_bg)
        self._thresh_marker.setFixedSize(2, 14)
        self._thresh_marker.setStyleSheet("background:#fafafa;border:none;")
        live.add(self._bar_bg)

        scale = QHBoxLayout()
        s0 = QLabel("0.00")
        s1 = QLabel("threshold 0.80")
        s2 = QLabel("1.00")
        for l in (s0, s1, s2):
            l.setFont(QFont(FONT_MONO, 7))
            l.setStyleSheet(f"color:{TXT3};background:transparent;border:none;")
        scale.addWidget(s0); scale.addStretch(); scale.addWidget(s1); scale.addStretch(); scale.addWidget(s2)
        live.add(scale)

        # Status badge
        self._verify_badge = QLabel("· awaiting frames ·")
        self._verify_badge.setFont(QFont(FONT_MONO, 10, QFont.Weight.Bold))
        self._verify_badge.setStyleSheet(
            f"color:{TXT3};background:rgba(255,255,255,0.04);"
            f"border:1px solid {BDR};border-radius:8px;padding:8px 14px;")
        self._verify_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        live.add(self._verify_badge)

        body.addWidget(live, 4)

        # Security log
        log_card = SectionCard("SECURITY LOG  ·  recent events")
        log_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._log_scroll = QScrollArea()
        self._log_scroll.setWidgetResizable(True)
        self._log_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._log_scroll.setStyleSheet(
            f"QScrollArea{{border:1px solid {BDR};border-radius:8px;background:#0f0f12;}}"
            "QScrollBar:vertical{width:6px;background:transparent;}"
            f"QScrollBar::handle:vertical{{background:#2a2a2e;border-radius:3px;}}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}")
        self._log_inner = QWidget(); self._log_inner.setStyleSheet("background:transparent;")
        self._log_lay = QVBoxLayout(self._log_inner)
        self._log_lay.setContentsMargins(8, 8, 8, 8); self._log_lay.setSpacing(2)
        self._log_lay.addStretch()
        self._log_scroll.setWidget(self._log_inner)
        log_card.add(self._log_scroll)

        body.addWidget(log_card, 5)

        # Initial state
        self._refresh_enrollment()
        self._refresh_log()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(800)

    # ─────────────────────────────────────────────────────────────────────
    def _refresh_enrollment(self):
        try:
            from modules.biometric_enroller import load_biometrics
        except Exception:
            return
        bio = load_biometrics(self._user_id)
        if bio is None:
            for t in (self._tile_face, self._tile_hand, self._tile_voice):
                t.set_state(False, "Enroll to enable identity verification.")
            return
        def _has(d, k):
            v = d.get(k)
            if v is None: return False
            try:
                import numpy as _np
                if isinstance(v, _np.ndarray): return v.size > 0
            except Exception: pass
            return bool(v)
        ts = bio.get("enrolled_at", "—")
        face_ok  = _has(bio, "face_ratios")
        hand_ok  = _has(bio, "hand_signature")
        voice_ok = bool(bio.get("voice_enabled", False))
        self._tile_face.set_state(face_ok,
            f"Enrolled {ts} · 20-float ratio + 1404-d embedding.")
        self._tile_hand.set_state(hand_ok,
            f"Enrolled {ts} · 15-float pose signature (5 poses).")
        self._tile_voice.set_state(voice_ok,
            f"Enrolled {ts} · 13-coeff MFCC voiceprint." if voice_ok
            else "Optional · enroll to enable voice match.")

    # ─────────────────────────────────────────────────────────────────────
    def _refresh_log(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "data",
            f"security_user{self._user_id}.csv")
        path = os.path.abspath(path)
        # Clear current rows
        while self._log_lay.count() > 1:
            it = self._log_lay.takeAt(0)
            if it.widget(): it.widget().deleteLater()
        if not os.path.exists(path):
            self._log_lay.insertWidget(0, self._empty_row("No security events recorded yet."))
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except Exception as e:
            self._log_lay.insertWidget(0, self._empty_row(f"Could not read log: {e}"))
            return
        rows = rows[-30:]
        if not rows:
            self._log_lay.insertWidget(0, self._empty_row("No events yet."))
            return
        for r in reversed(rows):
            self._log_lay.insertWidget(0, self._build_log_row(r))

    def _empty_row(self, text: str) -> QWidget:
        l = QLabel(text)
        l.setFont(QFont(FONT_MONO, 9))
        l.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l.setStyleSheet(f"color:{TXT3};background:transparent;border:none;padding:24px;")
        return l

    def _build_log_row(self, r: dict) -> QWidget:
        ev = (r.get("type") or "").upper()
        ts = r.get("t", "")
        score = r.get("score", "—")
        if ev == "IMPOSTOR":      col = "#ef4444"
        elif "FAIL" in ev:        col = "#fbbf24"
        elif ev == "RESUMED":     col = GREEN
        else:                     col = TXT2

        w = QFrame()
        w.setStyleSheet(
            f"QFrame{{background:transparent;border:none;border-bottom:1px solid {BDR};border-radius:0;}}")
        ly = QHBoxLayout(w)
        ly.setContentsMargins(8, 5, 8, 5); ly.setSpacing(8)
        # dot
        d = QFrame(); d.setFixedSize(6, 6)
        d.setStyleSheet(f"background:{col};border-radius:3px;border:none;")
        ly.addWidget(d)
        # ts
        tl = QLabel(ts); tl.setFont(QFont(FONT_MONO, 8))
        tl.setStyleSheet(f"color:{TXT3};background:transparent;border:none;")
        tl.setFixedWidth(140)
        ly.addWidget(tl)
        # type
        et = QLabel(ev); et.setFont(QFont(FONT_MONO, 9, QFont.Weight.Bold))
        et.setStyleSheet(f"color:{col};background:transparent;border:none;")
        et.setFixedWidth(110)
        ly.addWidget(et)
        # score
        sc = QLabel(f"score {score}")
        sc.setFont(QFont(FONT_MONO, 8))
        sc.setStyleSheet(f"color:{TXT2};background:transparent;border:none;")
        ly.addWidget(sc)
        ly.addStretch()
        return w

    # ─────────────────────────────────────────────────────────────────────
    def _tick(self):
        v = self._get_verifier()
        if v is None or not getattr(v, "enabled", False):
            self._verify_badge.setText("· verifier offline ·")
            self._verify_badge.setStyleSheet(
                f"color:{TXT3};background:rgba(255,255,255,0.04);"
                f"border:1px solid {BDR};border-radius:8px;padding:8px 14px;")
            return
        score = float(getattr(v, "last_score", 0.0))
        thr   = float(getattr(v, "face_threshold", 0.8))
        impostor = bool(getattr(v, "impostor_detected", False))

        self._score_value.setText(f"{score:.3f}")
        # Bar
        w = max(0, int(self._bar_bg.width() * max(0.0, min(1.0, score))))
        self._bar_fill.setGeometry(0, 0, w, 8)
        # Threshold marker
        tx = max(0, int(self._bar_bg.width() * thr) - 1)
        self._thresh_marker.setGeometry(tx, -3, 2, 14)
        # Color
        col = GREEN if score >= thr and not impostor else "#ef4444" if impostor else "#fbbf24"
        self._bar_fill.setStyleSheet(f"background:{col};border:none;border-radius:4px;")
        # Badge
        if impostor:
            self._verify_badge.setText("⚠  IMPOSTOR FLAGGED")
            self._verify_badge.setStyleSheet(
                f"color:#ef4444;background:rgba(239,68,68,0.1);"
                f"border:1px solid rgba(239,68,68,0.3);border-radius:8px;padding:8px 14px;")
        elif score >= thr:
            self._verify_badge.setText("✓  IDENTITY VERIFIED")
            self._verify_badge.setStyleSheet(
                f"color:{GREEN};background:rgba(34,197,94,0.1);"
                f"border:1px solid rgba(34,197,94,0.3);border-radius:8px;padding:8px 14px;")
        else:
            self._verify_badge.setText("…  VERIFYING")
            self._verify_badge.setStyleSheet(
                f"color:#fbbf24;background:rgba(251,191,36,0.1);"
                f"border:1px solid rgba(251,191,36,0.3);border-radius:8px;padding:8px 14px;")

        # Periodic log refresh (every ~5 ticks ≈ 4 s)
        if not hasattr(self, "_log_counter"): self._log_counter = 0
        self._log_counter += 1
        if self._log_counter >= 5:
            self._log_counter = 0
            self._refresh_log()
