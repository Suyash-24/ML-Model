"""
pages/settings_page.py
──────────────────────
Live config editor — writes to config/settings.json and emits a
`settings_saved(updates: dict)` signal so the host window can hot-reload
into running modules where safe.
"""

from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout,
    QScrollArea, QComboBox, QLineEdit, QPushButton, QSizePolicy,
)
from PyQt6.QtCore  import Qt, pyqtSignal, QTimer
from PyQt6.QtGui   import QFont

from ._common import (
    BG, PNL, PNL2, BDR, TXT, TXT2, TXT3, GREEN, ACCENT, ACCENT2,
    FONT, FONT_MONO,
    lbl, section, page_header,
    SectionCard, Toggle, LabeledSlider,
    control_row, primary_button, action_button,
)


_LINEEDIT_QSS = (
    f"QLineEdit{{background:#0f0f12;color:{TXT};"
    f"border:1px solid {BDR};border-radius:6px;padding:6px 10px;"
    f"font-family:'Consolas';font-size:10px;}}"
    f"QLineEdit:focus{{border:1px solid {ACCENT};}}"
)
_COMBO_QSS = (
    f"QComboBox{{background:#0f0f12;color:{TXT};"
    f"border:1px solid {BDR};border-radius:6px;padding:6px 10px;"
    f"font-family:'Consolas';font-size:10px;min-width:120px;}}"
    f"QComboBox:hover{{border:1px solid {ACCENT};}}"
    f"QComboBox::drop-down{{border:none;width:18px;}}"
    f"QComboBox QAbstractItemView{{background:#0f0f12;color:{TXT};"
    f"selection-background-color:{ACCENT};border:1px solid {BDR};}}"
)


class SettingsPage(QWidget):
    """Settings — full live editor."""
    settings_saved = pyqtSignal(dict)   # emits the dict of new values
    accent_changed = pyqtSignal(str)    # accent color hex

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self._cfg = config
        self._dirty: dict = {}
        self._controls = {}     # key -> control reference for read-back

        self.setStyleSheet(f"background:{BG};")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 16, 28, 12)
        outer.setSpacing(10)

        # Header
        head, self._badge = page_header(
            "⚙",
            "S  E  T  T  I  N  G  S",
            "Configure camera, modalities and biometrics. Changes save to "
            "config/settings.json — most apply live.",
            badge_text="● SAVED",
            badge_color=GREEN,
        )
        outer.addWidget(head)

        # Scrollable two-column body
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            "QScrollArea{border:none;background:transparent;}"
            "QScrollBar:vertical{width:6px;background:transparent;}"
            f"QScrollBar::handle:vertical{{background:#2a2a2e;border-radius:3px;}}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}")
        body = QWidget(); body.setStyleSheet("background:transparent;")
        grid = QGridLayout(body)
        grid.setContentsMargins(0, 0, 4, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(14)

        # ── Left column ──────────────────────────────────────────────────
        grid.addWidget(self._build_camera_card(),   0, 0)
        grid.addWidget(self._build_eye_card(),      1, 0)
        grid.addWidget(self._build_gesture_card(),  2, 0)

        # ── Right column ─────────────────────────────────────────────────
        grid.addWidget(self._build_voice_card(),    0, 1)
        grid.addWidget(self._build_bio_card(),      1, 1)
        grid.addWidget(self._build_appearance_card(), 2, 1)

        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        grid.setRowStretch(3, 1)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        # Footer action bar
        foot = QFrame()
        foot.setStyleSheet(
            f"QFrame{{background:{PNL2};border:1px solid {BDR};border-radius:10px;}}")
        fl = QHBoxLayout(foot)
        fl.setContentsMargins(16, 8, 16, 8); fl.setSpacing(10)

        self._dirty_lbl = QLabel("0 unsaved changes")
        self._dirty_lbl.setFont(QFont(FONT_MONO, 9))
        self._dirty_lbl.setStyleSheet(f"color:{TXT3};background:transparent;border:none;")
        fl.addWidget(self._dirty_lbl)
        fl.addStretch()

        self._reset_btn = action_button("⟲  Reset", TXT2)
        self._reset_btn.clicked.connect(self._reset_dirty)
        fl.addWidget(self._reset_btn)

        self._save_btn = primary_button("💾  Save changes", ACCENT)
        self._save_btn.clicked.connect(self._save)
        fl.addWidget(self._save_btn)
        outer.addWidget(foot)

    # ─────────────────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────────────────
    def _track(self, key: str, control):
        """Register a control under its config key."""
        self._controls[key] = control

    def _mark_dirty(self, key: str, value):
        if self._cfg.get(key) == value:
            self._dirty.pop(key, None)
        else:
            self._dirty[key] = value
        self._refresh_dirty_ui()

    def _refresh_dirty_ui(self):
        n = len(self._dirty)
        if n:
            self._dirty_lbl.setText(
                f"{n} unsaved change{'s' if n != 1 else ''}")
            self._dirty_lbl.setStyleSheet(
                f"color:#fbbf24;background:transparent;border:none;")
            if self._badge:
                self._badge.setText("  ● UNSAVED")
                self._badge.setStyleSheet(
                    f"color:#fbbf24;background:rgba(251,191,36,0.08);"
                    f"border:1px solid rgba(251,191,36,0.3);"
                    f"border-radius:6px;padding:0 12px;")
        else:
            self._dirty_lbl.setText("0 unsaved changes")
            self._dirty_lbl.setStyleSheet(
                f"color:{TXT3};background:transparent;border:none;")
            if self._badge:
                self._badge.setText("  ● SAVED")
                self._badge.setStyleSheet(
                    f"color:{GREEN};background:rgba(34,197,94,0.08);"
                    f"border:1px solid rgba(34,197,94,0.25);"
                    f"border-radius:6px;padding:0 12px;")

    def _reset_dirty(self):
        for key, ctrl in self._controls.items():
            if key not in self._dirty:
                continue
            current = self._cfg.get(key)
            if isinstance(ctrl, Toggle):
                ctrl.blockSignals(True); ctrl.setChecked(bool(current)); ctrl.blockSignals(False)
            elif isinstance(ctrl, LabeledSlider):
                # Re-create slider position
                ctrl.blockSignals(True)
                ctrl._slider.setValue(int(round(float(current) * ctrl._scale)))
                ctrl._refresh_label(float(current))
                ctrl.blockSignals(False)
            elif isinstance(ctrl, QLineEdit):
                ctrl.blockSignals(True); ctrl.setText(str(current or "")); ctrl.blockSignals(False)
            elif isinstance(ctrl, QComboBox):
                idx = ctrl.findData(current)
                ctrl.blockSignals(True)
                if idx >= 0: ctrl.setCurrentIndex(idx)
                ctrl.blockSignals(False)
        self._dirty.clear()
        self._refresh_dirty_ui()

    def _save(self):
        if not self._dirty:
            return
        # Coerce ints/floats per current config
        coerced = {}
        for k, v in self._dirty.items():
            cur = self._cfg.get(k)
            if isinstance(cur, bool):       coerced[k] = bool(v)
            elif isinstance(cur, int):      coerced[k] = int(v)
            elif isinstance(cur, float):    coerced[k] = float(v)
            else:                           coerced[k] = v
        self._cfg.set_many(coerced)
        try:
            self._cfg.save()
        except Exception as e:
            self._dirty_lbl.setText(f"Save failed: {e}")
            return
        self.settings_saved.emit(coerced)
        self._dirty.clear()
        self._refresh_dirty_ui()

    # ─────────────────────────────────────────────────────────────────────
    #  Card builders
    # ─────────────────────────────────────────────────────────────────────
    def _build_camera_card(self) -> SectionCard:
        c = SectionCard("📷  CAMERA")
        # Camera index
        cam_combo = QComboBox()
        cam_combo.setStyleSheet(_COMBO_QSS)
        for i in range(4):
            cam_combo.addItem(f"Camera {i}", i)
        idx = cam_combo.findData(int(self._cfg.get("camera_index", 0)))
        if idx >= 0: cam_combo.setCurrentIndex(idx)
        cam_combo.currentIndexChanged.connect(
            lambda _i, cb=cam_combo: self._mark_dirty("camera_index", cb.currentData()))
        self._track("camera_index", cam_combo)
        c.add(control_row("Camera index", cam_combo))

        for key, title, vmin, vmax, step, sfx in [
            ("cam_width",  "Width  (px)",  320, 1920, 16, "px"),
            ("cam_height", "Height (px)",  240, 1080, 16, "px"),
            ("cam_fps",    "Target FPS",    10, 60,    1, " fps"),
        ]:
            s = LabeledSlider(title, vmin, vmax, float(self._cfg.get(key, vmin)),
                              step=step, suffix=sfx)
            s.valueChanged.connect(lambda v, k=key: self._mark_dirty(k, v))
            self._track(key, s); c.add(s)

        hint = lbl("Camera changes apply on next launch.", 9, TXT3)
        c.add(hint)
        return c

    def _build_eye_card(self) -> SectionCard:
        c = SectionCard("👁  EYE TRACKING")
        for key, label in [
            ("blink_click",      "Blink to click"),
            ("dwell_click",      "Dwell to click"),
            ("eye_moves_cursor", "Eye moves cursor"),
            ("auto_calibrate",   "Auto-calibrate on launch"),
        ]:
            t = Toggle(bool(self._cfg.get(key, False)))
            t.toggled.connect(lambda v, k=key: self._mark_dirty(k, bool(v)))
            self._track(key, t)
            c.add(control_row(label, t))

        for key, title, vmin, vmax, step, dec in [
            ("gaze_smooth_frames", "Gaze smoothing frames",  3,  12, 1, 0),
            ("eye_inactive_secs",  "Inactivity timeout (s)", 5,  60, 1, 0),
        ]:
            v = float(self._cfg.get(key, vmin))
            s = LabeledSlider(title, vmin, vmax, v, step=step, decimals=dec)
            s.valueChanged.connect(lambda v, k=key: self._mark_dirty(k, v))
            self._track(key, s); c.add(s)
        return c

    def _build_gesture_card(self) -> SectionCard:
        c = SectionCard("✋  GESTURES")
        for key, title, vmin, vmax, step, dec, sfx in [
            ("gesture_detect_conf",     "Detection confidence",   0.3,  0.95, 0.01, 2, ""),
            ("gesture_track_conf",      "Tracking confidence",    0.3,  0.95, 0.01, 2, ""),
            ("gesture_confirm_frames",  "Confirm frames",         1,    20,   1,    0, ""),
            ("gesture_global_cooldown", "Global cooldown frames", 4,    40,   1,    0, ""),
            ("pinch_distance_thresh",   "Pinch distance",         0.03, 0.10, 0.005, 3, ""),
        ]:
            v = float(self._cfg.get(key, vmin))
            s = LabeledSlider(title, vmin, vmax, v, step=step, decimals=dec, suffix=sfx)
            s.valueChanged.connect(lambda v, k=key: self._mark_dirty(k, v))
            self._track(key, s); c.add(s)

        t = Toggle(bool(self._cfg.get("scroll_use_shift_hscroll", True)))
        t.toggled.connect(
            lambda v: self._mark_dirty("scroll_use_shift_hscroll", bool(v)))
        self._track("scroll_use_shift_hscroll", t)
        c.add(control_row("Use Shift+wheel for H-scroll", t))
        return c

    def _build_voice_card(self) -> SectionCard:
        c = SectionCard("🎙  WEDNESDAY  /  VOICE")
        # Wake word
        we = QLineEdit(str(self._cfg.get("wake_word", "wednesday")))
        we.setStyleSheet(_LINEEDIT_QSS)
        we.textChanged.connect(
            lambda t: self._mark_dirty("wake_word", t.strip().lower()))
        self._track("wake_word", we)
        c.add(control_row("Wake word", we))

        # Mic energy threshold
        s = LabeledSlider("Mic energy threshold", 100, 1500,
                          float(self._cfg.get("mic_energy_threshold", 300)),
                          step=10)
        s.valueChanged.connect(
            lambda v: self._mark_dirty("mic_energy_threshold", v))
        self._track("mic_energy_threshold", s); c.add(s)

        s2 = LabeledSlider("Memory turns retained", 0, 30,
                           float(self._cfg.get("wednesday_memory_turns", 10)),
                           step=1)
        s2.valueChanged.connect(
            lambda v: self._mark_dirty("wednesday_memory_turns", int(v)))
        self._track("wednesday_memory_turns", s2); c.add(s2)

        # Ollama model
        ol = QLineEdit(str(self._cfg.get("ollama_model", "llama3.2:3b")))
        ol.setStyleSheet(_LINEEDIT_QSS)
        ol.textChanged.connect(lambda t: self._mark_dirty("ollama_model", t.strip()))
        self._track("ollama_model", ol)
        c.add(control_row("Ollama model", ol))

        # ElevenLabs key (masked)
        el = QLineEdit(str(self._cfg.get("elevenlabs_api_key", "")))
        el.setEchoMode(QLineEdit.EchoMode.Password)
        el.setStyleSheet(_LINEEDIT_QSS)
        el.setPlaceholderText("optional — premium TTS")
        el.textChanged.connect(
            lambda t: self._mark_dirty("elevenlabs_api_key", t.strip()))
        self._track("elevenlabs_api_key", el)
        c.add(control_row("ElevenLabs API key", el))
        return c

    def _build_bio_card(self) -> SectionCard:
        c = SectionCard("🔒  BIOMETRICS")
        for key, title, vmin, vmax, step, dec in [
            ("bio_face_threshold",      "Face match threshold",   0.50, 0.95, 0.01, 2),
            ("bio_voice_threshold",     "Voice match threshold",  0.50, 0.95, 0.01, 2),
            ("bio_fail_streak_limit",   "Fail streak → impostor", 1,    10,   1,    0),
            ("bio_grace_period_secs",   "Grace period (s)",       0,    60,   1,    0),
            ("bio_check_interval_secs", "Verification interval (s)", 1, 30,   1,    0),
        ]:
            cur = float(self._cfg.get(key, vmin))
            s = LabeledSlider(title, vmin, vmax, cur, step=step, decimals=dec)
            s.valueChanged.connect(lambda v, k=key: self._mark_dirty(k, v))
            self._track(key, s); c.add(s)
        return c

    def _build_appearance_card(self) -> SectionCard:
        c = SectionCard("🎨  APPEARANCE")
        # Accent swatches
        c.add(lbl("Accent color", 9, TXT3, mono=True))

        row = QHBoxLayout(); row.setSpacing(10)
        accents = [
            ("Green",  "#22c55e"),
            ("Violet", "#8b5cf6"),
            ("Blue",   "#3b82f6"),
            ("Orange", "#f97316"),
        ]
        current = str(self._cfg.get("accent_color", "#22c55e"))
        for name, hexv in accents:
            sw = QPushButton(name)
            sw.setCursor(Qt.CursorShape.PointingHandCursor)
            sw.setFixedHeight(34)
            border = "2px solid #fafafa" if hexv.lower() == current.lower() else f"1px solid {BDR}"
            sw.setStyleSheet(
                f"QPushButton{{background:{hexv}22;color:{hexv};"
                f"border:{border};border-radius:8px;"
                f"font-family:'Consolas';font-size:10px;font-weight:bold;}}"
                f"QPushButton:hover{{background:{hexv}33;}}")
            sw.clicked.connect(lambda _, h=hexv: self._pick_accent(h))
            row.addWidget(sw)
        c.add(row)

        # Voice speaking duration display
        s = LabeledSlider("Voice speaking secs", 1, 8,
                          float(self._cfg.get("voice_speaking_secs", 2)),
                          step=1)
        s.valueChanged.connect(
            lambda v: self._mark_dirty("voice_speaking_secs", int(v)))
        self._track("voice_speaking_secs", s); c.add(s)

        # AI cooldown
        s2 = LabeledSlider("AI action cooldown (frames)", 0, 60,
                           float(self._cfg.get("ai_action_cooldown_frames", 15)),
                           step=1)
        s2.valueChanged.connect(
            lambda v: self._mark_dirty("ai_action_cooldown_frames", int(v)))
        self._track("ai_action_cooldown_frames", s2); c.add(s2)

        return c

    def _pick_accent(self, hex_color: str):
        self._mark_dirty("accent_color", hex_color)
        # Rebuild appearance card swatches by re-marking borders
        # (cheap: just emit a preview signal — full restyle on save)
        self.accent_changed.emit(hex_color)
