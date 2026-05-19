"""
pages/profile_page.py
─────────────────────
Account self-service: avatar header, account details, change password,
update email, biometric summary and sign-out.
"""

import os
import csv
from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout,
    QLineEdit, QPushButton, QSizePolicy,
)
from PyQt6.QtCore  import Qt, pyqtSignal
from PyQt6.QtGui   import QFont, QPainter, QColor, QPen, QBrush, QPainterPath

from ._common import (
    BG, PNL, PNL2, BDR, TXT, TXT2, TXT3, GREEN, ACCENT, ACCENT2,
    FONT, FONT_MONO, lbl, section, page_header,
    SectionCard, action_button, primary_button,
)


_LE_QSS = (
    f"QLineEdit{{background:#0f0f12;color:{TXT};"
    f"border:1px solid {BDR};border-radius:6px;padding:8px 12px;"
    f"font-family:'Consolas';font-size:11px;}}"
    f"QLineEdit:focus{{border:1px solid {ACCENT};}}"
)


class _Avatar(QFrame):
    """Initial-based circular avatar."""
    def __init__(self, initials: str, color: str = ACCENT, size: int = 80):
        super().__init__()
        self.setFixedSize(size, size)
        self.setStyleSheet("background:transparent;border:none;")
        self._initials = initials.upper()[:2]
        self._color = color

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect()
        # gradient circle
        p.setPen(Qt.PenStyle.NoPen)
        col = QColor(self._color); col.setAlpha(60)
        p.setBrush(col)
        p.drawEllipse(0, 0, r.width(), r.height())
        col2 = QColor(self._color); col2.setAlpha(160)
        p.setBrush(col2)
        p.drawEllipse(4, 4, r.width()-8, r.height()-8)
        # text
        p.setPen(QColor("#fafafa"))
        p.setFont(QFont(FONT, r.width() // 3, QFont.Weight.Bold))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, self._initials)
        p.end()


class ProfilePage(QWidget):
    sign_out_requested = pyqtSignal()
    navigate_to        = pyqtSignal(str)

    def __init__(self, user: dict, parent=None):
        super().__init__(parent)
        self._user = dict(user) if user else {}
        self.setStyleSheet(f"background:{BG};")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 16, 28, 12)
        outer.setSpacing(10)

        head, _ = page_header(
            "👤",
            "P  R  O  F  I  L  E",
            "Manage your account, password and biometric enrollments.",
            badge_text="",
        )
        outer.addWidget(head)

        body = QHBoxLayout(); body.setSpacing(14)
        outer.addLayout(body, 1)

        body.addWidget(self._build_left(),  5)
        body.addWidget(self._build_right(), 5)

    # ─────────────────────────────────────────────────────────────────────
    def _build_left(self) -> QWidget:
        w = QWidget(); w.setStyleSheet("background:transparent;")
        ly = QVBoxLayout(w); ly.setContentsMargins(0,0,0,0); ly.setSpacing(12)

        # ── Header card ──
        head = QFrame()
        head.setStyleSheet(
            f"QFrame{{background:{PNL2};border:1px solid {BDR};border-radius:14px;}}")
        head.setFixedHeight(140)
        hl = QHBoxLayout(head); hl.setContentsMargins(20, 16, 20, 16); hl.setSpacing(16)

        full = (self._user.get("full_name") or
                self._user.get("username") or "User")
        initials = "".join(p[0] for p in full.split()[:2]) or full[:2]
        avatar = _Avatar(initials, ACCENT, 84)
        hl.addWidget(avatar)

        col = QVBoxLayout(); col.setSpacing(2)
        nm = QLabel(full)
        nm.setFont(QFont(FONT, 18, QFont.Weight.Bold))
        nm.setStyleSheet(f"color:{TXT};background:transparent;border:none;")
        col.addWidget(nm)
        un = QLabel(f"@{self._user.get('username','—')}  ·  {self._user.get('email','—')}")
        un.setFont(QFont(FONT_MONO, 9))
        un.setStyleSheet(f"color:{TXT3};background:transparent;border:none;")
        col.addWidget(un)
        joined = (self._user.get("created_at") or "—")[:10]
        last   = (self._user.get("last_login") or "—")[:10]
        meta = QLabel(f"member since {joined}  ·  last login {last}")
        meta.setFont(QFont(FONT_MONO, 8))
        meta.setStyleSheet(f"color:{TXT3};background:transparent;border:none;")
        col.addWidget(meta)
        col.addStretch()
        hl.addLayout(col, 1)

        ly.addWidget(head)

        # ── Stats row ──
        stats = QHBoxLayout(); stats.setSpacing(10)
        # Compute from sessions.csv
        n_sessions, total_secs, total_cmds = self._compute_stats()
        if total_secs > 3600:
            t_str = f"{total_secs/3600:.1f}h"
        elif total_secs > 60:
            t_str = f"{total_secs/60:.0f}m"
        else:
            t_str = f"{int(total_secs)}s"
        for title, val, color in [
            ("Sessions",       str(n_sessions), ACCENT),
            ("Total time",     t_str,           ACCENT2),
            ("Commands fired", str(total_cmds), "#38bdf8"),
        ]:
            stats.addWidget(self._stat_card(title, val, color))
        ly.addLayout(stats)

        # ── Biometric summary ──
        bio = SectionCard("BIOMETRIC ENROLLMENT")
        try:
            from modules.biometric_enroller import load_biometrics
            uid = int(self._user.get("id", 0))
            data = load_biometrics(uid)
        except Exception:
            data = None

        def _has(d, k):
            if not d: return False
            v = d.get(k)
            if v is None: return False
            try:
                import numpy as _np
                if isinstance(v, _np.ndarray): return v.size > 0
            except Exception: pass
            return bool(v)

        for kind, label, key in [
            ("Face",  "👤  Face print",   "face_ratios"),
            ("Hand",  "✋  Hand pose",     "hand_signature"),
            ("Voice", "🎙  Voice MFCC",    "voice_enabled"),
        ]:
            bio.add(self._bio_row(label, _has(data, key)))

        manage = action_button("Manage biometrics →", ACCENT2)
        manage.clicked.connect(lambda: self.navigate_to.emit("Biometrics"))
        bio.add(manage)
        ly.addWidget(bio)
        ly.addStretch()
        return w

    def _stat_card(self, title: str, value: str, color: str) -> QFrame:
        c = QFrame()
        c.setStyleSheet(
            f"QFrame{{background:{PNL2};border:1px solid {BDR};border-radius:10px;}}")
        c.setFixedHeight(78)
        ly = QVBoxLayout(c); ly.setContentsMargins(14, 10, 14, 10); ly.setSpacing(2)
        t = QLabel(title.upper())
        t.setFont(QFont(FONT_MONO, 7))
        t.setStyleSheet(
            f"color:{TXT3};background:transparent;border:none;letter-spacing:2px;")
        ly.addWidget(t)
        v = QLabel(value)
        v.setFont(QFont(FONT, 18, QFont.Weight.Bold))
        v.setStyleSheet(f"color:{color};background:transparent;border:none;")
        ly.addWidget(v)
        return c

    def _bio_row(self, label: str, ok: bool) -> QWidget:
        row = QFrame()
        row.setStyleSheet("background:transparent;border:none;")
        rl = QHBoxLayout(row); rl.setContentsMargins(0,0,0,0); rl.setSpacing(8)
        l = QLabel(label)
        l.setFont(QFont(FONT, 10))
        l.setStyleSheet(f"color:{TXT};background:transparent;border:none;")
        rl.addWidget(l, 1)
        pill = QLabel("  ● ENROLLED  " if ok else "  ○ NONE  ")
        pill.setFont(QFont(FONT_MONO, 8, QFont.Weight.Bold))
        pill.setFixedHeight(22)
        if ok:
            pill.setStyleSheet(
                f"color:{GREEN};background:rgba(34,197,94,0.1);"
                f"border:1px solid rgba(34,197,94,0.3);border-radius:11px;padding:0 8px;")
        else:
            pill.setStyleSheet(
                f"color:{TXT3};background:rgba(255,255,255,0.04);"
                f"border:1px solid {BDR};border-radius:11px;padding:0 8px;")
        rl.addWidget(pill)
        return row

    # ─────────────────────────────────────────────────────────────────────
    def _build_right(self) -> QWidget:
        w = QWidget(); w.setStyleSheet("background:transparent;")
        ly = QVBoxLayout(w); ly.setContentsMargins(0,0,0,0); ly.setSpacing(12)

        # ── Change password ──
        pw = SectionCard("CHANGE PASSWORD")
        self._pw_current = QLineEdit(); self._pw_current.setEchoMode(QLineEdit.EchoMode.Password)
        self._pw_current.setPlaceholderText("Current password")
        self._pw_new     = QLineEdit(); self._pw_new.setEchoMode(QLineEdit.EchoMode.Password)
        self._pw_new.setPlaceholderText("New password (min 4 chars)")
        self._pw_confirm = QLineEdit(); self._pw_confirm.setEchoMode(QLineEdit.EchoMode.Password)
        self._pw_confirm.setPlaceholderText("Confirm new password")
        for le in (self._pw_current, self._pw_new, self._pw_confirm):
            le.setStyleSheet(_LE_QSS)
            pw.add(le)
        self._pw_msg = QLabel("")
        self._pw_msg.setFont(QFont(FONT_MONO, 9))
        self._pw_msg.setStyleSheet(f"color:{TXT3};background:transparent;border:none;")
        pw.add(self._pw_msg)
        pw_btn_row = QHBoxLayout()
        pw_btn_row.addStretch()
        pw_btn = primary_button("Update password", ACCENT)
        pw_btn.clicked.connect(self._do_change_password)
        pw_btn_row.addWidget(pw_btn)
        pw.add(pw_btn_row)
        ly.addWidget(pw)

        # ── Update email ──
        em = SectionCard("EMAIL")
        self._em_input = QLineEdit(self._user.get("email", ""))
        self._em_input.setStyleSheet(_LE_QSS)
        em.add(self._em_input)
        self._em_msg = QLabel("")
        self._em_msg.setFont(QFont(FONT_MONO, 9))
        self._em_msg.setStyleSheet(f"color:{TXT3};background:transparent;border:none;")
        em.add(self._em_msg)
        em_btn_row = QHBoxLayout()
        em_btn_row.addStretch()
        em_btn = primary_button("Update email", ACCENT)
        em_btn.clicked.connect(self._do_update_email)
        em_btn_row.addWidget(em_btn)
        em.add(em_btn_row)
        ly.addWidget(em)

        # ── Sign out ──
        so = SectionCard("SESSION")
        sub = QLabel("Signing out will end the current session and return to the login screen.")
        sub.setFont(QFont(FONT, 9))
        sub.setWordWrap(True)
        sub.setStyleSheet(f"color:{TXT3};background:transparent;border:none;")
        so.add(sub)
        so_btn_row = QHBoxLayout(); so_btn_row.addStretch()
        so_btn = action_button("⏻  Sign out", "#ef4444")
        so_btn.clicked.connect(self.sign_out_requested.emit)
        so_btn_row.addWidget(so_btn)
        so.add(so_btn_row)
        ly.addWidget(so)

        ly.addStretch()
        return w

    # ─────────────────────────────────────────────────────────────────────
    def _do_change_password(self):
        cur = self._pw_current.text()
        new = self._pw_new.text()
        cnf = self._pw_confirm.text()
        if not cur or not new:
            return self._set_pw_msg("Fill all fields.", "#fbbf24")
        if new != cnf:
            return self._set_pw_msg("New passwords don't match.", "#ef4444")
        try:
            from auth import update_password
        except Exception:
            return self._set_pw_msg("Auth module unavailable.", "#ef4444")
        ok, msg = update_password(int(self._user.get("id", 0)), cur, new)
        self._set_pw_msg(msg, GREEN if ok else "#ef4444")
        if ok:
            self._pw_current.clear(); self._pw_new.clear(); self._pw_confirm.clear()

    def _set_pw_msg(self, text: str, color: str):
        self._pw_msg.setText(text)
        self._pw_msg.setStyleSheet(f"color:{color};background:transparent;border:none;")

    def _do_update_email(self):
        try:
            from auth import update_email
        except Exception:
            self._em_msg.setText("Auth module unavailable.")
            return
        new_e = self._em_input.text().strip()
        ok, msg = update_email(int(self._user.get("id", 0)), new_e)
        self._em_msg.setText(msg)
        self._em_msg.setStyleSheet(
            f"color:{GREEN if ok else '#ef4444'};background:transparent;border:none;")
        if ok:
            self._user["email"] = new_e

    # ─────────────────────────────────────────────────────────────────────
    def _compute_stats(self):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "sessions.csv")
        path = os.path.abspath(path)
        uid  = str(self._user.get("id", ""))
        if not os.path.exists(path) or not uid:
            return 0, 0, 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f)
                        if str(r.get("user_id","")) == uid]
        except Exception:
            return 0, 0, 0
        n = len(rows)
        secs = sum(int(r.get("duration_secs", 0) or 0) for r in rows)
        cmds = sum(int(r.get("total_commands", 0) or 0) for r in rows)
        return n, secs, cmds
