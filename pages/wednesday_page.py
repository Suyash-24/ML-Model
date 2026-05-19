"""
pages/wednesday_page.py
───────────────────────
WEDNESDAY AI — Mission Control.
Capabilities matrix · live command log · model/voice/wake config ·
memory viewer · test-command input.  No chat duplication.
"""

from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout,
    QScrollArea, QLineEdit, QPushButton, QSizePolicy, QApplication,
)
from PyQt6.QtCore  import Qt, pyqtSignal, QTimer
from PyQt6.QtGui   import QFont

from ._common import (
    BG, PNL, PNL2, BDR, TXT, TXT2, TXT3, GREEN, ACCENT, ACCENT2,
    FONT, FONT_MONO,
    lbl, section, page_header,
    SectionCard, action_button, primary_button,
)


# ── Capabilities matrix (mirrors the WEDNESDAY system prompt) ──────────────
_CAPABILITIES = [
    ("Apps", ACCENT, [
        ("open_app",       "Launch application",       "open chrome"),
        ("close_app",      "Close application",         "close chrome"),
        ("close_window",   "Alt+F4 active window",      "close window"),
        ("new_tab",        "New browser tab",           "open a new tab"),
        ("switch_tab",     "Cycle browser tab",         "switch tab"),
        ("open_url",       "Visit URL",                 "go to youtube"),
    ]),
    ("Media", ACCENT2, [
        ("media_play",     "Play / pause",              "play music"),
        ("media_next",     "Next track",                "next song"),
        ("media_prev",     "Previous track",            "previous track"),
        ("volume_up",      "Volume up",                 "louder"),
        ("volume_down",    "Volume down",               "quieter"),
        ("mute",           "Mute",                      "mute"),
        ("brightness_up",  "Brighten screen",           "increase brightness"),
        ("brightness_down","Dim screen",                "decrease brightness"),
    ]),
    ("Files", "#38bdf8", [
        ("open_file",      "Open file by path",         "open the report"),
        ("open_folder",    "Open Explorer folder",      "open downloads"),
        ("create_folder",  "Make a new folder",         "make folder Photos"),
        ("screenshot",     "Save a screenshot",         "take a screenshot"),
    ]),
    ("System", "#f59e0b", [
        ("tell_time",      "What time is it",           "what's the time"),
        ("tell_date",      "Today's date",              "what's today"),
        ("tell_weather",   "Weather lookup",            "weather in mumbai"),
        ("calculate",      "Math expression",           "what's 17 times 23"),
        ("lock_screen",    "Lock the workstation",      "lock my pc"),
        ("sleep_pc",       "Sleep the PC",              "go to sleep"),
        ("shutdown_pc",    "Shut down the PC",          "shut down"),
    ]),
    ("Mouse / Keyboard", "#a78bfa", [
        ("click",          "Click anywhere",            "click here"),
        ("scroll",         "Scroll page",               "scroll down"),
        ("hotkey",         "Send hotkey",               "press ctrl+t"),
        ("type_text",      "Type a string",             "type hello world"),
        ("select_all",     "Select all",                "select all"),
        ("copy",           "Copy",                      "copy that"),
        ("paste",          "Paste",                     "paste"),
        ("undo",           "Undo",                      "undo"),
        ("redo",           "Redo",                      "redo"),
        ("save",           "Save (Ctrl+S)",             "save the file"),
        ("find",           "Find in page",              "find error"),
    ]),
    ("Clipboard", "#fb923c", [
        ("get_clipboard",  "Read clipboard",            "what's on clipboard"),
        ("set_clipboard",  "Write to clipboard",        "copy this text"),
    ]),
    ("Eyecon Control", GREEN, [
        ("pause_system",   "Pause Eyecon",              "pause eyecon"),
        ("resume_system",  "Resume Eyecon",             "wake up eyecon"),
        ("eye_enable",     "Toggle eye control",        "activate eye control"),
        ("calibrate_eye",  "Run eye calibration",       "calibrate my eyes"),
    ]),
]


class _CategoryDot(QFrame):
    def __init__(self, color: str):
        super().__init__()
        self.setFixedSize(8, 8)
        self.setStyleSheet(f"background:{color};border-radius:4px;border:none;")


class _CapabilityRow(QPushButton):
    def __init__(self, action: str, desc: str, example: str, color: str):
        super().__init__()
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(46)
        self._example = example
        self.setStyleSheet(
            f"QPushButton{{text-align:left;background:transparent;border:none;"
            f"border-bottom:1px solid {BDR};border-radius:0;padding:6px 8px;}}"
            f"QPushButton:hover{{background:rgba(255,255,255,0.025);}}")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4); lay.setSpacing(10)

        dot = _CategoryDot(color)
        lay.addWidget(dot)

        col = QVBoxLayout(); col.setSpacing(0)
        nm = QLabel(action)
        nm.setFont(QFont(FONT_MONO, 9, QFont.Weight.Bold))
        nm.setStyleSheet(f"color:{TXT};background:transparent;border:none;")
        col.addWidget(nm)
        ds = QLabel(desc)
        ds.setFont(QFont(FONT, 9))
        ds.setStyleSheet(f"color:{TXT3};background:transparent;border:none;")
        col.addWidget(ds)
        lay.addLayout(col, 1)

        ex = QLabel(f"“{example}”")
        ex.setFont(QFont(FONT_MONO, 8))
        ex.setStyleSheet(f"color:{color};background:transparent;border:none;")
        lay.addWidget(ex)


class WednesdayMissionControlPage(QWidget):
    """Mission control surface for the WEDNESDAY engine."""
    test_command_submitted = pyqtSignal(str)
    test_voice_requested   = pyqtSignal()

    def __init__(self, config, engine_getter, parent=None):
        """engine_getter — callable returning the live WednesdayEngine or None."""
        super().__init__(parent)
        self._cfg = config
        self._get_engine = engine_getter
        self._log_entries = []     # most recent first
        self.setStyleSheet(f"background:{BG};")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 16, 28, 12)
        outer.setSpacing(10)

        head, self._badge = page_header(
            "✦",
            "W  E  D  N  E  S  D  A  Y    ·    M I S S I O N    C O N T R O L",
            "Capability surface, live action log, configuration and memory.",
            badge_text="● ENGINE READY",
            badge_color=GREEN,
        )
        outer.addWidget(head)

        # ── Body: 3 columns ─────────────────────────────────────────────
        body = QHBoxLayout(); body.setSpacing(14)
        outer.addLayout(body, 1)

        body.addWidget(self._build_capabilities_column(), 4)
        body.addWidget(self._build_log_column(),          4)
        body.addWidget(self._build_config_column(),       3)

        # Periodic engine-status refresh
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_status)
        self._refresh_timer.start(1500)

    # ─────────────────────────────────────────────────────────────────────
    #  CAPABILITIES
    # ─────────────────────────────────────────────────────────────────────
    def _build_capabilities_column(self) -> QWidget:
        col = QWidget(); col.setStyleSheet("background:transparent;")
        cl = QVBoxLayout(col); cl.setContentsMargins(0,0,0,0); cl.setSpacing(8)

        cl.addWidget(section("CAPABILITIES"))
        sub = QLabel(f"{sum(len(g[2]) for g in _CAPABILITIES)} actions across "
                     f"{len(_CAPABILITIES)} categories · click to copy example")
        sub.setFont(QFont(FONT, 9))
        sub.setStyleSheet(f"color:{TXT3};background:transparent;")
        cl.addWidget(sub)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea{{border:1px solid {BDR};border-radius:10px;background:{PNL2};}}"
            "QScrollBar:vertical{width:6px;background:transparent;}"
            f"QScrollBar::handle:vertical{{background:#2a2a2e;border-radius:3px;}}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}")
        inner = QWidget(); inner.setStyleSheet("background:transparent;")
        ly = QVBoxLayout(inner); ly.setContentsMargins(8, 8, 8, 8); ly.setSpacing(2)

        for category, color, rows in _CAPABILITIES:
            head = QHBoxLayout(); head.setSpacing(8)
            head.setContentsMargins(0, 8, 0, 4)
            head.addWidget(_CategoryDot(color))
            cl_lbl = QLabel(category.upper())
            cl_lbl.setFont(QFont(FONT_MONO, 9, QFont.Weight.Bold))
            cl_lbl.setStyleSheet(f"color:{color};background:transparent;letter-spacing:2px;border:none;")
            head.addWidget(cl_lbl)
            head.addStretch()
            count = QLabel(f"{len(rows)}")
            count.setFont(QFont(FONT_MONO, 8))
            count.setStyleSheet(f"color:{TXT3};background:transparent;border:none;")
            head.addWidget(count)
            head_w = QWidget(); head_w.setLayout(head)
            head_w.setStyleSheet("background:transparent;")
            ly.addWidget(head_w)

            for action, desc, example in rows:
                row = _CapabilityRow(action, desc, example, color)
                row.clicked.connect(lambda _, e=example: self._copy_example(e))
                ly.addWidget(row)

        ly.addStretch()
        scroll.setWidget(inner)
        cl.addWidget(scroll, 1)
        return col

    def _copy_example(self, example: str):
        QApplication.clipboard().setText(example)
        # brief flash of badge
        if self._badge:
            old = self._badge.text()
            self._badge.setText(f"  ⎘ COPIED  '{example[:28]}'")
            QTimer.singleShot(1400, lambda: self._badge.setText(old))

    # ─────────────────────────────────────────────────────────────────────
    #  LIVE LOG
    # ─────────────────────────────────────────────────────────────────────
    def _build_log_column(self) -> QWidget:
        col = QWidget(); col.setStyleSheet("background:transparent;")
        cl = QVBoxLayout(col); cl.setContentsMargins(0,0,0,0); cl.setSpacing(8)

        head_row = QHBoxLayout(); head_row.setSpacing(6)
        head_row.addWidget(section("LIVE COMMAND LOG"))
        head_row.addStretch()
        clr = QPushButton("clear")
        clr.setFont(QFont(FONT_MONO, 8))
        clr.setCursor(Qt.CursorShape.PointingHandCursor)
        clr.setStyleSheet(
            f"QPushButton{{color:{TXT3};background:transparent;border:none;"
            f"padding:0 6px;}}"
            f"QPushButton:hover{{color:{TXT};}}")
        clr.clicked.connect(self._clear_log)
        head_row.addWidget(clr)
        cl.addLayout(head_row)

        self._log_box = QFrame()
        self._log_box.setStyleSheet(
            f"QFrame{{background:{PNL2};border:1px solid {BDR};border-radius:10px;}}")
        lb = QVBoxLayout(self._log_box)
        lb.setContentsMargins(0, 0, 0, 0); lb.setSpacing(0)

        self._log_scroll = QScrollArea()
        self._log_scroll.setWidgetResizable(True)
        self._log_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._log_scroll.setStyleSheet(
            "QScrollArea{border:none;background:transparent;}"
            "QScrollBar:vertical{width:6px;background:transparent;}"
            f"QScrollBar::handle:vertical{{background:#2a2a2e;border-radius:3px;}}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}")
        self._log_inner = QWidget(); self._log_inner.setStyleSheet("background:transparent;")
        self._log_lay = QVBoxLayout(self._log_inner)
        self._log_lay.setContentsMargins(8, 8, 8, 8); self._log_lay.setSpacing(2)
        self._empty_lbl = QLabel("· no actions yet ·")
        self._empty_lbl.setFont(QFont(FONT_MONO, 9))
        self._empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_lbl.setStyleSheet(f"color:{TXT3};background:transparent;padding:24px;")
        self._log_lay.addWidget(self._empty_lbl)
        self._log_lay.addStretch()
        self._log_scroll.setWidget(self._log_inner)
        lb.addWidget(self._log_scroll)
        cl.addWidget(self._log_box, 1)

        # Test command bar
        test_row = QFrame()
        test_row.setStyleSheet(
            f"QFrame{{background:{PNL2};border:1px solid {BDR};border-radius:10px;}}")
        tr = QHBoxLayout(test_row)
        tr.setContentsMargins(8, 6, 8, 6); tr.setSpacing(6)
        self._test_input = QLineEdit()
        self._test_input.setPlaceholderText('test command e.g. "open chrome"')
        self._test_input.setFont(QFont(FONT_MONO, 10))
        self._test_input.setStyleSheet(
            f"QLineEdit{{background:#0f0f12;color:{TXT};border:1px solid {BDR};"
            f"border-radius:6px;padding:6px 10px;}}"
            f"QLineEdit:focus{{border:1px solid {ACCENT};}}")
        self._test_input.returnPressed.connect(self._submit_test)
        tr.addWidget(self._test_input, 1)
        run_btn = primary_button("▶  Run", ACCENT)
        run_btn.clicked.connect(self._submit_test)
        tr.addWidget(run_btn)
        cl.addWidget(test_row)
        return col

    def _submit_test(self):
        text = self._test_input.text().strip()
        if text:
            self._test_input.clear()
            self.test_command_submitted.emit(text)

    def _clear_log(self):
        self._log_entries.clear()
        # Remove all children except empty label + stretch
        while self._log_lay.count() > 2:
            item = self._log_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._empty_lbl.show()

    def add_log_entry(self, entry: dict):
        """Called from main window when WEDNESDAY emits a history event."""
        self._log_entries.insert(0, entry)
        if len(self._log_entries) > 200:
            self._log_entries = self._log_entries[:200]
        self._empty_lbl.hide()

        action = entry.get("action", "?")
        params = entry.get("params") or {}
        ok = entry.get("ok", True)
        ts = entry.get("t", "")

        row = QFrame()
        row.setStyleSheet(
            f"QFrame{{background:transparent;border:none;"
            f"border-bottom:1px solid {BDR};border-radius:0;}}")
        rl = QHBoxLayout(row)
        rl.setContentsMargins(6, 4, 6, 4); rl.setSpacing(8)

        # Time
        tl = QLabel(ts)
        tl.setFont(QFont(FONT_MONO, 8))
        tl.setStyleSheet(f"color:{TXT3};background:transparent;border:none;")
        tl.setFixedWidth(58)
        rl.addWidget(tl)

        # Status dot
        col = "#22c55e" if ok else "#ef4444"
        rl.addWidget(_CategoryDot(col))

        # Action name
        a_lbl = QLabel(action)
        a_lbl.setFont(QFont(FONT_MONO, 9, QFont.Weight.Bold))
        a_lbl.setStyleSheet(f"color:{TXT};background:transparent;border:none;")
        rl.addWidget(a_lbl)

        # Params summary
        psum = ", ".join(f"{k}={str(v)[:18]}" for k, v in list(params.items())[:3])
        if psum:
            ps = QLabel(psum)
            ps.setFont(QFont(FONT_MONO, 8))
            ps.setStyleSheet(f"color:{TXT3};background:transparent;border:none;")
            ps.setWordWrap(False)
            rl.addWidget(ps, 1)
        else:
            rl.addStretch()

        # Insert at top (just above existing entries; index 1 to skip empty_lbl... wait,
        # actually layout has empty_lbl at 0, stretch at end. After hide() empty_lbl
        # is still in layout but invisible — keep it that way and insert at index 1)
        self._log_lay.insertWidget(1, row)

        # Trim DOM (keep ~80 visible rows for perf)
        if self._log_lay.count() > 84:
            item = self._log_lay.takeAt(self._log_lay.count() - 2)  # -1 is stretch
            if item and item.widget():
                item.widget().deleteLater()

    # ─────────────────────────────────────────────────────────────────────
    #  CONFIG
    # ─────────────────────────────────────────────────────────────────────
    def _build_config_column(self) -> QWidget:
        col = QWidget(); col.setStyleSheet("background:transparent;")
        cl = QVBoxLayout(col); cl.setContentsMargins(0,0,0,0); cl.setSpacing(10)

        # MODEL
        c = SectionCard("LLM  ·  OLLAMA")
        self._model_lbl = QLabel(self._cfg.get("ollama_model", "llama3.2:3b"))
        self._model_lbl.setFont(QFont(FONT_MONO, 11, QFont.Weight.Bold))
        self._model_lbl.setStyleSheet(f"color:{TXT};background:transparent;border:none;")
        c.add(self._model_lbl)
        h = QLabel("Configure model in Settings · runs locally")
        h.setFont(QFont(FONT, 9))
        h.setStyleSheet(f"color:{TXT3};background:transparent;border:none;")
        c.add(h)
        cl.addWidget(c)

        # WAKE
        c2 = SectionCard("WAKE WORD")
        self._wake_lbl = QLabel(f"“{self._cfg.get('wake_word', 'wednesday')}”")
        self._wake_lbl.setFont(QFont(FONT_MONO, 11, QFont.Weight.Bold))
        self._wake_lbl.setStyleSheet(f"color:{TXT};background:transparent;border:none;")
        c2.add(self._wake_lbl)
        self._wake_state = QLabel("· checking engine ·")
        self._wake_state.setFont(QFont(FONT_MONO, 8))
        self._wake_state.setStyleSheet(f"color:{TXT3};background:transparent;border:none;")
        c2.add(self._wake_state)
        cl.addWidget(c2)

        # VOICE
        c3 = SectionCard("VOICE OUTPUT")
        key = self._cfg.get("elevenlabs_api_key", "")
        voice = QLabel("ElevenLabs (premium)" if key else "Windows Zira (free)")
        voice.setFont(QFont(FONT, 10))
        voice.setStyleSheet(f"color:{TXT};background:transparent;border:none;")
        c3.add(voice)
        tv = action_button("🔊  Test voice", ACCENT)
        tv.clicked.connect(self.test_voice_requested.emit)
        c3.add(tv)
        cl.addWidget(c3)

        # MEMORY
        c4 = SectionCard("MEMORY")
        self._mem_count = QLabel("0 turns persisted")
        self._mem_count.setFont(QFont(FONT_MONO, 9))
        self._mem_count.setStyleSheet(f"color:{TXT};background:transparent;border:none;")
        c4.add(self._mem_count)

        self._mem_box = QScrollArea()
        self._mem_box.setWidgetResizable(True)
        self._mem_box.setFixedHeight(160)
        self._mem_box.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._mem_box.setStyleSheet(
            f"QScrollArea{{border:1px solid {BDR};border-radius:8px;background:#0f0f12;}}"
            "QScrollBar:vertical{width:6px;background:transparent;}"
            f"QScrollBar::handle:vertical{{background:#2a2a2e;border-radius:3px;}}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}")
        self._mem_inner = QWidget(); self._mem_inner.setStyleSheet("background:transparent;")
        self._mem_lay = QVBoxLayout(self._mem_inner)
        self._mem_lay.setContentsMargins(8, 8, 8, 8); self._mem_lay.setSpacing(6)
        self._mem_lay.addStretch()
        self._mem_box.setWidget(self._mem_inner)
        c4.add(self._mem_box)

        mem_btns = QHBoxLayout(); mem_btns.setSpacing(8)
        rb = action_button("↻  Reload",  TXT2)
        rb.clicked.connect(self._reload_memory)
        cb = action_button("✕  Clear",   "#ef4444")
        cb.clicked.connect(self._clear_memory_clicked)
        mem_btns.addWidget(rb); mem_btns.addWidget(cb); mem_btns.addStretch()
        c4.add(mem_btns)

        cl.addWidget(c4)
        cl.addStretch()
        return col

    # ─────────────────────────────────────────────────────────────────────
    #  PUBLIC HOOKS
    # ─────────────────────────────────────────────────────────────────────
    def _refresh_status(self):
        engine = self._get_engine()
        if engine is None:
            if self._badge:
                self._badge.setText("  ● ENGINE OFFLINE")
                self._badge.setStyleSheet(
                    f"color:#ef4444;background:rgba(239,68,68,0.08);"
                    f"border:1px solid rgba(239,68,68,0.25);"
                    f"border-radius:6px;padding:0 12px;")
            self._wake_state.setText("· engine not running ·")
            return
        # Engine running
        if self._badge:
            self._badge.setText("  ● ENGINE READY")
            self._badge.setStyleSheet(
                f"color:{GREEN};background:rgba(34,197,94,0.08);"
                f"border:1px solid rgba(34,197,94,0.25);"
                f"border-radius:6px;padding:0 12px;")
        is_paused = bool(getattr(engine, "paused", False))
        if is_paused:
            self._wake_state.setText("· paused — wake disabled ·")
        else:
            self._wake_state.setText("· listening ·")

    def _reload_memory(self):
        engine = self._get_engine()
        if engine is None:
            return
        turns = engine.load_persisted_memory()
        # Clear current list
        while self._mem_lay.count() > 1:
            it = self._mem_lay.takeAt(0)
            if it.widget(): it.widget().deleteLater()
        self._mem_count.setText(f"{len(turns)} turn{'s' if len(turns) != 1 else ''} persisted")
        for turn in turns[-30:]:
            self._mem_lay.insertWidget(self._mem_lay.count() - 1,
                                       self._build_mem_row(turn))

    def _build_mem_row(self, turn: dict) -> QWidget:
        w = QFrame()
        w.setStyleSheet(
            f"QFrame{{background:{PNL2};border:1px solid {BDR};border-radius:6px;}}")
        ly = QVBoxLayout(w)
        ly.setContentsMargins(8, 6, 8, 6); ly.setSpacing(2)
        u = QLabel(f"YOU  ·  {turn.get('t','')[-8:]}")
        u.setFont(QFont(FONT_MONO, 7))
        u.setStyleSheet(f"color:{TXT3};background:transparent;border:none;letter-spacing:1px;")
        ly.addWidget(u)
        u_msg = QLabel((turn.get("user","") or "")[:140])
        u_msg.setFont(QFont(FONT, 9))
        u_msg.setWordWrap(True)
        u_msg.setStyleSheet(f"color:{TXT};background:transparent;border:none;")
        ly.addWidget(u_msg)
        a = QLabel("WEDNESDAY")
        a.setFont(QFont(FONT_MONO, 7))
        a.setStyleSheet(f"color:{ACCENT2};background:transparent;border:none;letter-spacing:1px;")
        ly.addWidget(a)
        a_msg = QLabel((turn.get("wednesday","") or "")[:200])
        a_msg.setFont(QFont(FONT, 9))
        a_msg.setWordWrap(True)
        a_msg.setStyleSheet(f"color:{TXT2};background:transparent;border:none;")
        ly.addWidget(a_msg)
        return w

    def _clear_memory_clicked(self):
        engine = self._get_engine()
        if engine is None: return
        engine.clear_memory()
        self._reload_memory()
