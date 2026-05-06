"""
auth_window.py  —  Eyecon Login / Sign-Up Native Window
────────────────────────────────────────────────────────
Full PyQt6 native application window with:
  • Login page  (username/email + password)
  • Sign-up page (multi-field form with useful data collection)
  • Animated transitions between pages
  • Matching dark Eyecon aesthetic
  • Field validation with inline error messages
  • "Remember me" token (local file)
  • Data collected: name, email, age group, use-case, disability flag, country
"""

import os
import json
import sys

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFrame, QComboBox, QCheckBox, QStackedWidget,
    QScrollArea, QSizePolicy, QApplication, QMainWindow,
    QGraphicsOpacityEffect,
)
from PyQt6.QtCore  import Qt, QPropertyAnimation, QEasingCurve, QTimer, pyqtSignal
from PyQt6.QtGui   import QFont, QColor, QPixmap, QPainter, QLinearGradient, QBrush

from utils.auth import register_user, login_user, init_db
from biometric_window import BiometricEnrollmentWindow

# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg":           "#0d1117",
    "panel":        "#161b22",
    "panel2":       "#1c2128",
    "border":       "#21262d",
    "border_hi":    "#30363d",
    "accent":       "#58a6ff",
    "accent_dark":  "#1f3a5f",
    "green":        "#3fb950",
    "green_dark":   "#0f2a1a",
    "red":          "#f85149",
    "red_dark":     "#3d1a1a",
    "yellow":       "#e3b341",
    "text":         "#e6edf3",
    "text_sec":     "#8b949e",
    "text_ter":     "#484f58",
    "input_bg":     "#0d1117",
    "input_border": "#30363d",
    "input_focus":  "#58a6ff",
    "placeholder":  "#484f58",
}

REMEMBER_FILE = os.path.join(os.path.dirname(__file__), "data", ".remember")


# ─────────────────────────────────────────────────────────────────────────────
#  STYLED COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
def _font(size=11, bold=False, mono=True):
    family = "Consolas" if mono else "Segoe UI"
    w = QFont.Weight.Bold if bold else QFont.Weight.Normal
    return QFont(family, size, w)


def _lbl(text, size=11, color=C["text"], bold=False, mono=True):
    l = QLabel(text)
    l.setFont(_font(size, bold, mono))
    l.setStyleSheet(f"color:{color}; background:transparent;")
    return l


class EyeInput(QLineEdit):
    """Styled input field with focus highlight."""
    def __init__(self, placeholder="", password=False, parent=None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        if password:
            self.setEchoMode(QLineEdit.EchoMode.Password)
        self.setFixedHeight(42)
        self.setFont(_font(11))
        self._set_normal()

    def _set_normal(self):
        self.setStyleSheet(f"""
            QLineEdit {{
                background: {C['input_bg']};
                color: {C['text']};
                border: 1px solid {C['input_border']};
                border-radius: 6px;
                padding: 0 12px;
                font-family: Consolas;
                font-size: 11px;
            }}
            QLineEdit:focus {{
                border: 1px solid {C['accent']};
            }}
            QLineEdit::placeholder {{
                color: {C['placeholder']};
            }}
        """)

    def set_error(self, on: bool):
        border = C["red"] if on else C["input_border"]
        self.setStyleSheet(f"""
            QLineEdit {{
                background: {C['input_bg']};
                color: {C['text']};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 0 12px;
                font-family: Consolas;
                font-size: 11px;
            }}
            QLineEdit:focus {{
                border: 1px solid {C['accent'] if not on else C['red']};
            }}
        """)


class EyeCombo(QComboBox):
    def __init__(self, options, parent=None):
        super().__init__(parent)
        self.setFixedHeight(42)
        self.setFont(_font(11))
        for opt in options:
            self.addItem(opt)
        self.setStyleSheet(f"""
            QComboBox {{
                background: {C['input_bg']};
                color: {C['text']};
                border: 1px solid {C['input_border']};
                border-radius: 6px;
                padding: 0 12px;
                font-family: Consolas;
                font-size: 11px;
            }}
            QComboBox:focus {{ border: 1px solid {C['accent']}; }}
            QComboBox::drop-down {{
                border: none;
                padding-right: 10px;
            }}
            QComboBox QAbstractItemView {{
                background: {C['panel2']};
                color: {C['text']};
                border: 1px solid {C['border_hi']};
                selection-background-color: {C['accent_dark']};
                font-family: Consolas;
            }}
        """)


class PrimaryButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFixedHeight(44)
        self.setFont(_font(11, bold=True))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"""
            QPushButton {{
                background: {C['accent']};
                color: #0d1117;
                border: none;
                border-radius: 6px;
                font-family: Consolas;
                font-size: 11px;
                font-weight: bold;
                letter-spacing: 1px;
            }}
            QPushButton:hover  {{ background: #79b8ff; }}
            QPushButton:pressed {{ background: #388bfd; }}
            QPushButton:disabled {{ background: {C['border']}; color: {C['text_ter']}; }}
        """)


class GhostButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFixedHeight(44)
        self.setFont(_font(11))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {C['accent']};
                border: 1px solid {C['border_hi']};
                border-radius: 6px;
                font-family: Consolas;
                font-size: 11px;
            }}
            QPushButton:hover  {{ background: {C['accent_dark']}; border-color: {C['accent']}; }}
            QPushButton:pressed {{ background: #0d1f3c; }}
        """)


class ErrorBanner(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(_font(10))
        self.setWordWrap(True)
        self.setFixedHeight(36)
        self.hide()

    def show_error(self, msg):
        self.setText(f"  ✕  {msg}")
        self.setStyleSheet(f"""
            color: {C['red']};
            background: {C['red_dark']};
            border: 1px solid {C['red']}55;
            border-radius: 6px;
            padding: 0 8px;
        """)
        self.show()

    def show_success(self, msg):
        self.setText(f"  ✓  {msg}")
        self.setStyleSheet(f"""
            color: {C['green']};
            background: {C['green_dark']};
            border: 1px solid {C['green']}55;
            border-radius: 6px;
            padding: 0 8px;
        """)
        self.show()

    def clear_msg(self):
        self.hide()


class Divider(QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(1)
        self.setStyleSheet(f"background: {C['border']}; border: none;")


# ─────────────────────────────────────────────────────────────────────────────
#  LEFT BRANDING PANEL
# ─────────────────────────────────────────────────────────────────────────────
class BrandPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(340)
        self.setStyleSheet(f"background: {C['panel']}; border-right: 1px solid {C['border']};")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(36, 48, 36, 36)
        lay.setSpacing(0)

        # Logo
        logo = QLabel()
        logo.setTextFormat(Qt.TextFormat.RichText)
        logo.setText(
            f'<span style="font-family:Consolas; font-size:28px; font-weight:bold; color:{C["accent"]}">EYE</span>'
            f'<span style="font-family:Consolas; font-size:28px; font-weight:bold; color:{C["text"]}">CON</span>'
        )
        lay.addWidget(logo)
        lay.addSpacing(6)

        ver = _lbl("v2.1.0", size=9, color=C["text_ter"])
        lay.addWidget(ver)
        lay.addSpacing(32)

        tagline = QLabel("AI-Powered\nMultimodal\nInteraction\nSystem")
        tagline.setFont(QFont("Consolas", 16, QFont.Weight.Bold))
        tagline.setStyleSheet(f"color: {C['text']}; background: transparent;")
        tagline.setWordWrap(True)
        lay.addWidget(tagline)
        lay.addSpacing(20)

        sub = QLabel(
            "Control your computer with\n"
            "your eyes, hands, and voice.\n"
            "No keyboard. No mouse."
        )
        sub.setFont(QFont("Consolas", 10))
        sub.setStyleSheet(f"color: {C['text_sec']}; background: transparent;")
        lay.addWidget(sub)
        lay.addSpacing(48)

        # Feature chips
        for icon, feat in [
            ("👀", "Eye Tracking & Gaze Control"),
            ("✋", "Hand Gesture Recognition"),
            ("🎙️", "Voice Command Engine"),
            ("🧠", "AI Decision Module"),
        ]:
            row = QHBoxLayout()
            row.setSpacing(10)
            ic = QLabel(icon)
            ic.setFont(QFont("Segoe UI Emoji", 14))
            ic.setFixedWidth(28)
            ic.setStyleSheet("background: transparent;")
            ft = _lbl(feat, size=10, color=C["text_sec"])
            row.addWidget(ic)
            row.addWidget(ft)
            row.addStretch()
            lay.addLayout(row)
            lay.addSpacing(10)

        lay.addStretch()

        # Bottom note
        note = _lbl("Your data is stored locally.\nWe never sell your information.",
                     size=9, color=C["text_ter"])
        note.setWordWrap(True)
        lay.addWidget(note)


# ─────────────────────────────────────────────────────────────────────────────
#  LOGIN PAGE
# ─────────────────────────────────────────────────────────────────────────────
class LoginPage(QWidget):
    login_success = pyqtSignal(dict)   # emits user dict
    go_signup     = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background: {C['bg']};")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }"
                             "QScrollBar { width: 0px; }")
        outer.addWidget(scroll)

        inner = QWidget()
        inner.setStyleSheet("background: transparent;")
        scroll.setWidget(inner)

        lay = QVBoxLayout(inner)
        lay.setContentsMargins(56, 56, 56, 56)
        lay.setSpacing(0)

        lay.addWidget(_lbl("Welcome back", size=22, bold=True))
        lay.addSpacing(6)
        lay.addWidget(_lbl("Sign in to your Eyecon account", size=11, color=C["text_sec"]))
        lay.addSpacing(32)

        self._banner = ErrorBanner()
        lay.addWidget(self._banner)
        lay.addSpacing(4)

        # Fields
        lay.addWidget(_lbl("Username or Email", size=10, color=C["text_sec"]))
        lay.addSpacing(4)
        self._user_input = EyeInput("johndoe or john@email.com")
        lay.addWidget(self._user_input)
        lay.addSpacing(16)

        lay.addWidget(_lbl("Password", size=10, color=C["text_sec"]))
        lay.addSpacing(4)
        self._pass_input = EyeInput("••••••••", password=True)
        lay.addWidget(self._pass_input)
        lay.addSpacing(10)

        # Remember me
        self._remember = QCheckBox("Keep me signed in")
        self._remember.setFont(_font(10))
        self._remember.setStyleSheet(f"""
            QCheckBox {{ color: {C['text_sec']}; background: transparent; spacing: 8px; }}
            QCheckBox::indicator {{
                width: 16px; height: 16px;
                border: 1px solid {C['input_border']};
                border-radius: 3px;
                background: {C['input_bg']};
            }}
            QCheckBox::indicator:checked {{
                background: {C['accent']};
                border-color: {C['accent']};
            }}
        """)
        lay.addWidget(self._remember)
        lay.addSpacing(24)

        self._btn_login = PrimaryButton("SIGN IN")
        self._btn_login.clicked.connect(self._do_login)
        lay.addWidget(self._btn_login)
        lay.addSpacing(16)

        # Divider
        div_row = QHBoxLayout()
        div_row.addWidget(Divider(), 1)
        div_row.addWidget(_lbl("  or  ", size=9, color=C["text_ter"]), 0)
        div_row.addWidget(Divider(), 1)
        lay.addLayout(div_row)
        lay.addSpacing(16)

        self._btn_signup = GhostButton("CREATE ACCOUNT")
        self._btn_signup.clicked.connect(self.go_signup.emit)
        lay.addWidget(self._btn_signup)
        lay.addStretch()

        # Enter key triggers login
        self._pass_input.returnPressed.connect(self._do_login)
        self._user_input.returnPressed.connect(self._pass_input.setFocus)

        # Auto-fill from remember file
        self._load_remember()

    def _do_login(self):
        self._banner.clear_msg()
        username = self._user_input.text().strip()
        password = self._pass_input.text()

        if not username:
            self._user_input.set_error(True)
            self._banner.show_error("Please enter your username or email.")
            return
        if not password:
            self._pass_input.set_error(True)
            self._banner.show_error("Please enter your password.")
            return

        self._user_input.set_error(False)
        self._pass_input.set_error(False)
        self._btn_login.setEnabled(False)
        self._btn_login.setText("Signing in…")

        # Slight delay for UX feel
        QTimer.singleShot(400, lambda: self._attempt_login(username, password))

    def _attempt_login(self, username, password):
        ok, result = login_user(username, password)
        self._btn_login.setEnabled(True)
        self._btn_login.setText("SIGN IN")

        if ok:
            if self._remember.isChecked():
                self._save_remember(username)
            self._banner.show_success(f"Welcome back, {result.get('username', '')}!")
            QTimer.singleShot(600, lambda: self.login_success.emit(result))
        else:
            self._banner.show_error(result)
            self._pass_input.set_error(True)

    def _save_remember(self, username):
        os.makedirs(os.path.dirname(REMEMBER_FILE), exist_ok=True)
        with open(REMEMBER_FILE, "w") as f:
            json.dump({"username": username}, f)

    def _load_remember(self):
        if os.path.exists(REMEMBER_FILE):
            try:
                with open(REMEMBER_FILE) as f:
                    data = json.load(f)
                self._user_input.setText(data.get("username", ""))
                self._remember.setChecked(True)
                self._pass_input.setFocus()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
#  SIGN-UP PAGE
# ─────────────────────────────────────────────────────────────────────────────
class SignupPage(QWidget):
    signup_success = pyqtSignal(dict)
    go_login       = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background: {C['bg']};")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }"
                             "QScrollBar { width: 0px; }")
        outer.addWidget(scroll)

        inner = QWidget()
        inner.setStyleSheet("background: transparent;")
        scroll.setWidget(inner)

        lay = QVBoxLayout(inner)
        lay.setContentsMargins(56, 44, 56, 44)
        lay.setSpacing(0)

        lay.addWidget(_lbl("Create your account", size=20, bold=True))
        lay.addSpacing(4)
        lay.addWidget(_lbl("Help us personalise Eyecon for you", size=11, color=C["text_sec"]))
        lay.addSpacing(24)

        self._banner = ErrorBanner()
        lay.addWidget(self._banner)
        lay.addSpacing(4)

        # ── Section: Account ──────────────────────────────────────────
        lay.addWidget(self._section_label("ACCOUNT"))
        lay.addSpacing(8)

        row1 = QHBoxLayout(); row1.setSpacing(12)
        self._full_name = EyeInput("Full name  (optional)")
        self._username  = EyeInput("Username  (lowercase, 3–20 chars)*")
        row1.addWidget(self._full_name, 1)
        row1.addWidget(self._username, 1)
        lay.addLayout(row1)
        lay.addSpacing(2)
        uname_hint = _lbl("Letters, digits, underscores, periods. Must start with a letter.",
                          size=8, color=C["text_ter"])
        lay.addWidget(uname_hint)
        lay.addSpacing(10)

        self._email = EyeInput("Email address*")
        lay.addWidget(self._email)
        lay.addSpacing(10)

        row2 = QHBoxLayout(); row2.setSpacing(12)
        self._password  = EyeInput("Password  (min 8 chars)*", password=True)
        self._password2 = EyeInput("Confirm password*",        password=True)
        row2.addWidget(self._password, 1)
        row2.addWidget(self._password2, 1)
        lay.addLayout(row2)
        lay.addSpacing(2)
        pw_hint = _lbl("Must include uppercase, lowercase, digit, and special character.",
                       size=8, color=C["text_ter"])
        lay.addWidget(pw_hint)
        lay.addSpacing(24)

        # ── Section: Profile (research data) ──────────────────────────
        lay.addWidget(self._section_label("YOUR PROFILE"))
        lay.addWidget(_lbl("This helps us improve Eyecon for everyone.",
                           size=9, color=C["text_ter"]))
        lay.addSpacing(10)

        row3 = QHBoxLayout(); row3.setSpacing(12)
        self._age_group = EyeCombo([
            "Age group (optional)",
            "Under 18",
            "18 – 25",
            "26 – 35",
            "36 – 50",
            "50+",
        ])
        self._country = EyeInput("Country  (optional)")
        row3.addWidget(self._age_group, 1)
        row3.addWidget(self._country, 1)
        lay.addLayout(row3)
        lay.addSpacing(10)

        row4 = QHBoxLayout(); row4.setSpacing(12)
        self._use_case = EyeCombo([
            "Why are you using Eyecon?",
            "Accessibility / disability aid",
            "Productivity / hands-free work",
            "Research / university project",
            "Gaming",
            "Just exploring",
            "Other",
        ])
        self._disability = EyeCombo([
            "Mobility / disability (optional)",
            "Motor impairment",
            "Visual impairment",
            "None",
            "Prefer not to say",
        ])
        row4.addWidget(self._use_case, 1)
        row4.addWidget(self._disability, 1)
        lay.addLayout(row4)
        lay.addSpacing(24)

        # ── Section: Preferences ──────────────────────────────────────
        lay.addWidget(self._section_label("INTERACTION PREFERENCES"))
        lay.addSpacing(8)

        self._pref_mode = EyeCombo([
            "Default interaction mode",
            "Multimodal (all three)",
            "Eye tracking only",
            "Gesture only",
            "Voice only",
        ])
        lay.addWidget(self._pref_mode)
        lay.addSpacing(16)

        # Consent
        self._consent = QCheckBox(
            "I agree to store my usage data locally to improve my experience"
        )
        self._consent.setFont(_font(9))
        self._consent.setChecked(True)
        self._consent.setStyleSheet(f"""
            QCheckBox {{ color: {C['text_sec']}; background: transparent; spacing: 8px; }}
            QCheckBox::indicator {{
                width: 14px; height: 14px;
                border: 1px solid {C['input_border']};
                border-radius: 3px;
                background: {C['input_bg']};
            }}
            QCheckBox::indicator:checked {{
                background: {C['accent']};
                border-color: {C['accent']};
            }}
        """)
        lay.addWidget(self._consent)
        lay.addSpacing(20)

        self._btn_create = PrimaryButton("CREATE ACCOUNT")
        self._btn_create.clicked.connect(self._do_signup)
        lay.addWidget(self._btn_create)
        lay.addSpacing(16)

        div_row = QHBoxLayout()
        div_row.addWidget(Divider(), 1)
        div_row.addWidget(_lbl("  or  ", size=9, color=C["text_ter"]), 0)
        div_row.addWidget(Divider(), 1)
        lay.addLayout(div_row)
        lay.addSpacing(16)

        btn_back = GhostButton("ALREADY HAVE AN ACCOUNT? SIGN IN")
        btn_back.clicked.connect(self.go_login.emit)
        lay.addWidget(btn_back)
        lay.addSpacing(8)

    def _section_label(self, text):
        lbl = _lbl(text, size=9, color=C["text_ter"])
        lbl.setStyleSheet(
            f"color: {C['text_ter']}; background: transparent;"
            f" border-bottom: 1px solid {C['border']}; padding-bottom: 6px;"
        )
        return lbl

    def _age_val(self):
        v = self._age_group.currentText()
        return "" if "optional" in v else v

    def _use_val(self):
        v = self._use_case.currentText()
        return "" if "?" in v else v

    def _dis_val(self):
        v = self._disability.currentText()
        return "" if "optional" in v else v

    def _mode_val(self):
        v = self._pref_mode.currentText()
        mapping = {
            "Multimodal (all three)": "MULTI",
            "Eye tracking only":      "EYE",
            "Gesture only":           "GESTURE",
            "Voice only":             "VOICE",
        }
        return mapping.get(v, "MULTI")

    def _do_signup(self):
        self._banner.clear_msg()

        # Clear errors
        for f in [self._username, self._email, self._password, self._password2]:
            f.set_error(False)

        username  = self._username.text().strip()
        email     = self._email.text().strip()
        password  = self._password.text()
        password2 = self._password2.text()
        full_name = self._full_name.text().strip()

        # ── Quick client-side checks (full validation in backend) ─────
        if not username:
            self._username.set_error(True)
            self._banner.show_error("Username is required.")
            return
        if not email:
            self._email.set_error(True)
            self._banner.show_error("Email address is required.")
            return
        if not password:
            self._password.set_error(True)
            self._banner.show_error("Password is required.")
            return
        if password != password2:
            self._password.set_error(True)
            self._password2.set_error(True)
            self._banner.show_error("Passwords do not match.")
            return

        self._btn_create.setEnabled(False)
        self._btn_create.setText("Creating account…")

        QTimer.singleShot(500, lambda: self._attempt_signup(
            username, email, password, full_name,
            self._age_val(), self._use_val(), self._dis_val(),
            self._country.text().strip(),
        ))

    def _attempt_signup(self, username, email, password, full_name,
                        age_group, use_case, disability, country):
        ok, result = register_user(
            username, email, password,
            full_name=full_name,
            age_group=age_group,
            use_case=use_case,
            disability=disability,
            country=country,
        )
        self._btn_create.setEnabled(True)
        self._btn_create.setText("CREATE ACCOUNT")

        if ok:
            # result is now a full user dict from register_user
            user = result
            user["_is_new"] = True      # mark as new user for biometric enrollment
            self._banner.show_success(f"Welcome, {user.get('username', '')}! Signing you in…")
            QTimer.singleShot(800, lambda: self.signup_success.emit(user))
        else:
            # result is an error message string — highlight the relevant field
            err = result.lower()
            if "username" in err:
                self._username.set_error(True)
            elif "email" in err:
                self._email.set_error(True)
            elif "password" in err:
                self._password.set_error(True)
            elif "name" in err:
                self._full_name.set_error(True)
            self._banner.show_error(result)


# ─────────────────────────────────────────────────────────────────────────────
#  AUTH WINDOW  (Login ↔ Signup with animated transition)
# ─────────────────────────────────────────────────────────────────────────────
class AuthWindow(QMainWindow):
    authenticated = pyqtSignal(dict)   # emitted when user successfully logs in

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eyecon — Sign In")
        self.setFixedSize(900, 640)
        self._center()

        init_db()

        root = QWidget()
        root.setStyleSheet(f"background: {C['bg']};")
        self.setCentralWidget(root)

        main_lay = QHBoxLayout(root)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)

        # Left branding
        main_lay.addWidget(BrandPanel())

        # Right: stacked login / signup
        self._stack = QStackedWidget()
        self._stack.setStyleSheet(f"background: {C['bg']};")
        main_lay.addWidget(self._stack, 1)

        self._login_page  = LoginPage()
        self._signup_page = SignupPage()
        self._stack.addWidget(self._login_page)   # index 0
        self._stack.addWidget(self._signup_page)  # index 1

        # Connections
        self._login_page.go_signup.connect(lambda: self._switch(1))
        self._signup_page.go_login.connect(lambda: self._switch(0))
        self._login_page.login_success.connect(self._on_auth)
        self._signup_page.signup_success.connect(self._on_auth)

    def _switch(self, index):
        """Fade transition between pages."""
        effect = QGraphicsOpacityEffect(self._stack.currentWidget())
        self._stack.currentWidget().setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity")
        anim.setDuration(180)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.finished.connect(lambda: self._finish_switch(index))
        anim.start()
        self._anim = anim  # keep reference

    def _finish_switch(self, index):
        self._stack.setCurrentIndex(index)
        w = self._stack.currentWidget()
        effect = QGraphicsOpacityEffect(w)
        w.setGraphicsEffect(effect)
        anim2 = QPropertyAnimation(effect, b"opacity")
        anim2.setDuration(200)
        anim2.setStartValue(0.0)
        anim2.setEndValue(1.0)
        anim2.setEasingCurve(QEasingCurve.Type.InCubic)
        anim2.start()
        self._anim2 = anim2

    def _on_auth(self, user: dict):
        """Called on successful login OR after signup auto-login."""
        is_new_user = user.get("_is_new", False)

        if is_new_user:
            # Show biometric enrollment before launching dashboard
            self._bio_win = BiometricEnrollmentWindow(
                user_id=user["id"],
                username=user["username"]
            )
            self._bio_win.enrolled.connect(
                lambda uid: self._after_enrollment(user)
            )
            self._bio_win.show()
            self.hide()
        else:
            self._after_enrollment(user)

    def _after_enrollment(self, user: dict):
        self.authenticated.emit(user)
        self.close()

    def _center(self):
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width()  - 900) // 2
        y = (screen.height() - 640) // 2
        self.move(x, y)


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QToolTip { background: #161b22; color: #e6edf3;
                   border: 1px solid #30363d; font-family: Consolas; }
    """)
    win = AuthWindow()
    win.authenticated.connect(lambda u: print(f"\n✅ Authenticated: {u['username']}"))
    win.show()
    sys.exit(app.exec())