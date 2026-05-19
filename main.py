r"""
 _____ _  _ ___ ___  ___  _  _
| __| || | __/ __|/ _ \| \| |
| _| \_, | _|(__| (_) | .` |
|___| |_/|___\___|\\___/|_|\\_|

Eyecon — AI-Powered Multimodal Interaction System
Author: Eyecon Team
Version: 2.1.0
"""

"""
main.py  —  Eyecon Entry Point
───────────────────────────────
Flow:
  1. Show auth window (login / signup)
  2. If new user → biometric enrollment
  3. Launch FRIDAY main window

PLACE AT: ML-Model/main.py  (replace existing)
"""

import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore    import Qt
# QtWebEngineWidgets MUST be imported before QApplication is created
from PyQt6.QtWebEngineWidgets import QWebEngineView  # noqa: F401


def main():
    # High-DPI support — must be set before QApplication is created
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app.setApplicationName("Eyecon FRIDAY")
    app.setApplicationVersion("2.1.0")

    # Global stylesheet -- minimal, dark
    app.setStyleSheet("""
        QToolTip {
            background: #141c2b;
            color: #e2e8f0;
            border: 1px solid rgba(255,255,255,0.08);
            font-family: 'Consolas';
            font-size: 10px;
        }
        QScrollBar:vertical {
            background: transparent;
            width: 4px;
        }
        QScrollBar::handle:vertical {
            background: #1e293b;
            border-radius: 2px;
            min-height: 20px;
        }
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical { height: 0px; }
    """)

    # ── Step 1: Auth window ───────────────────────────────────────────────────
    from auth_window import AuthWindow
    auth_win = AuthWindow()

    main_win_ref = [None]

    def on_authenticated(user: dict):
        """Called when login/signup succeeds."""
        is_new = user.pop("_is_new", False)

        if is_new:
            # Step 2: Biometric enrollment for new users
            try:
                from biometric_window import BiometricEnrollmentWindow
                bio_win = BiometricEnrollmentWindow(
                    user_id=user.get("id", 1),
                    username=user.get("username", "user")
                )

                def on_enrolled(uid):
                    _launch_main(user)

                bio_win.enrolled.connect(on_enrolled)
                bio_win.show()
                auth_win._bio_win = bio_win  # keep reference
            except ImportError:
                _launch_main(user)
        else:
            _launch_main(user)

    def _launch_main(user: dict):
        """Step 3: Launch FRIDAY main window."""
        from app import EyeconWindow
        win = EyeconWindow(user=user)
        main_win_ref[0] = win
        win.show()

    auth_win.authenticated.connect(on_authenticated)
    auth_win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()