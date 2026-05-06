"""
biometric_window.py  —  Eyecon Biometric Enrollment UI
───────────────────────────────────────────────────────
PyQt6 native window shown ONCE after successful sign-up.

Flow:
  Step 0 — Welcome + explanation
  Step 1 — Face capture   (live camera, progress bar, 30 frames)
  Step 2 — Hand capture   (5 poses × 10 frames, guided instructions)
  Step 3 — Voice capture  (3 phrases, countdown timer)
  Step 4 — Done / summary

All captures run in a background thread; UI updates via Qt signals.
On completion → emits enrolled(user_id) signal → main app launches.
User can click "Skip" to bypass (biometrics saved as empty vectors).
"""

import sys
import time
import threading
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QFrame, QApplication, QMainWindow, QSizePolicy,
    QStackedWidget,
)
from PyQt6.QtCore  import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui   import QFont, QPixmap, QImage, QColor

# ─── colours (match Eyecon theme) ────────────────────────────────────────────
C = {
    "bg":      "#0d1117",
    "panel":   "#161b22",
    "border":  "#21262d",
    "accent":  "#58a6ff",
    "green":   "#3fb950",
    "yellow":  "#e3b341",
    "red":     "#f85149",
    "text":    "#e6edf3",
    "sec":     "#8b949e",
    "ter":     "#484f58",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Qt signal bridge (runs capture in thread, emits to UI)
# ─────────────────────────────────────────────────────────────────────────────
class EnrollSignals(QObject):
    progress     = pyqtSignal(str, int, int, object, str)  # (step, done, total, frame, label)
    step_done    = pyqtSignal(str)                          # step name
    all_done     = pyqtSignal(int)                          # user_id
    error        = pyqtSignal(str)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _lbl(text, size=11, color=C["text"], bold=False):
    l = QLabel(text)
    l.setFont(QFont("Consolas", size,
                    QFont.Weight.Bold if bold else QFont.Weight.Normal))
    l.setStyleSheet(f"color:{color}; background:transparent;")
    l.setWordWrap(True)
    return l


def _bar(color=C["accent"]):
    pb = QProgressBar()
    pb.setFixedHeight(5)
    pb.setTextVisible(False)
    pb.setRange(0, 100)
    pb.setStyleSheet(
        f"QProgressBar {{ background:{C['border']}; border-radius:2px; border:none; }}"
        f"QProgressBar::chunk {{ background:{color}; border-radius:2px; }}"
    )
    return pb


def _frame_to_pixmap(frame, w=320, h=240):
    """Convert OpenCV BGR frame to QPixmap."""
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h_, w_ = rgb.shape[:2]
    img   = QImage(rgb.data, w_, h_,
                   w_ * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img).scaled(
        w, h,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  STEP PAGES
# ─────────────────────────────────────────────────────────────────────────────
class WelcomePage(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 30, 40, 30)
        lay.setSpacing(14)

        icon = QLabel("🔐")
        icon.setFont(QFont("Segoe UI Emoji", 36))
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet("background:transparent;")
        lay.addWidget(icon)
        lay.addSpacing(6)

        lay.addWidget(_lbl("Set up biometric security",
                           size=18, bold=True))
        lay.addSpacing(4)
        lay.addWidget(_lbl(
            "Eyecon will learn your face, hand geometry, and voice so only "
            "you can control the system. This takes about 60 seconds.",
            size=11, color=C["sec"]
        ))
        lay.addSpacing(20)

        for icon_txt, title, desc in [
            ("👀", "Face recognition",
             "30 frames captured • stored as a 1404-float embedding"),
            ("✋", "Hand geometry",
             "5 hand poses • 15-float ratio signature (pose-invariant)"),
            ("🎙️", "Voice print",
             "3 short phrases • 13 MFCC coefficients averaged"),
        ]:
            row = QHBoxLayout(); row.setSpacing(12)
            ic = QLabel(icon_txt)
            ic.setFont(QFont("Segoe UI Emoji", 18))
            ic.setFixedWidth(32)
            ic.setStyleSheet("background:transparent;")
            col = QVBoxLayout(); col.setSpacing(2)
            col.addWidget(_lbl(title, size=11, bold=True))
            col.addWidget(_lbl(desc,  size=9, color=C["ter"]))
            row.addWidget(ic)
            row.addLayout(col, 1)
            lay.addLayout(row)

        lay.addSpacing(14)
        lay.addWidget(_lbl(
            "No raw video or audio is saved — only maths vectors.",
            size=9, color=C["ter"]
        ))
        lay.addStretch()


class CapturePage(QWidget):
    def __init__(self, step_title, instructions):
        super().__init__()
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(30, 20, 30, 20)
        lay.setSpacing(10)

        lay.addWidget(_lbl(step_title, size=15, bold=True))

        self._instr = _lbl(instructions, size=10, color=C["sec"])
        lay.addWidget(self._instr)
        lay.addSpacing(4)

        # Camera feed
        self._cam_lbl = QLabel()
        self._cam_lbl.setFixedSize(380, 260)
        self._cam_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cam_lbl.setStyleSheet(
            f"background:{C['bg']}; border:1px solid {C['border']}; border-radius:8px;"
        )
        cam_wrap = QHBoxLayout()
        cam_wrap.addStretch()
        cam_wrap.addWidget(self._cam_lbl)
        cam_wrap.addStretch()
        lay.addLayout(cam_wrap)

        # Status label
        self._status = _lbl("Ready…", size=10, color=C["sec"])
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._status)

        # Progress bar
        self._bar_label = _lbl("", size=9, color=C["ter"])
        lay.addWidget(self._bar_label)
        self._bar = _bar()
        lay.addWidget(self._bar)

        # Pose instruction for hand page
        self._pose_lbl = _lbl("", size=13, color=C["yellow"], bold=True)
        self._pose_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._pose_lbl)

        lay.addStretch()

    def update_progress(self, done, total, frame=None, extra_label=""):
        pct = int(done / total * 100) if total else 0
        self._bar.setValue(pct)
        self._bar_label.setText(f"{done} / {total} frames captured")
        self._status.setText(f"Capturing… {pct}%")
        self._pose_lbl.setText(extra_label)
        if frame is not None:
            self._cam_lbl.setPixmap(_frame_to_pixmap(frame))

    def set_done(self):
        self._bar.setValue(100)
        self._status.setText("✓  Captured successfully")
        self._status.setStyleSheet(f"color:{C['green']}; background:transparent;")
        self._pose_lbl.setText("")

    def set_instruction(self, text):
        self._instr.setText(text)


class VoicePage(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(30, 20, 30, 20)
        lay.setSpacing(10)

        lay.addWidget(_lbl("Voice print capture", size=15, bold=True))
        lay.addWidget(_lbl("Say each phrase clearly when the countdown starts.",
                           size=10, color=C["sec"]))
        lay.addSpacing(10)

        self._phrase_lbl = _lbl("", size=18, bold=True)
        self._phrase_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._phrase_lbl)
        lay.addSpacing(8)

        self._status = _lbl("Waiting…", size=11, color=C["sec"])
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._status)
        lay.addSpacing(8)

        self._bar = _bar(C["yellow"])
        lay.addWidget(self._bar)

        self._progress_lbl = _lbl("", size=9, color=C["ter"])
        lay.addWidget(self._progress_lbl)
        lay.addStretch()

    def set_phrase(self, phrase, phrase_num, total):
        self._phrase_lbl.setText(f'"{phrase}"')
        self._progress_lbl.setText(f"Phrase {phrase_num+1} of {total}")
        self._bar.setValue(int(phrase_num / total * 100))

    def set_status(self, text, color=None):
        self._status.setText(text)
        c = color or C["sec"]
        self._status.setStyleSheet(f"color:{c}; background:transparent;")

    def set_done(self):
        self._bar.setValue(100)
        self._phrase_lbl.setText("Voice captured!")
        self._status.setText("✓  Voice print saved")
        self._status.setStyleSheet(f"color:{C['green']}; background:transparent;")


class DonePage(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 40, 40, 40)
        lay.setSpacing(16)
        lay.addStretch()

        icon = QLabel("✅")
        icon.setFont(QFont("Segoe UI Emoji", 40))
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet("background:transparent;")
        lay.addWidget(icon)

        lay.addWidget(_lbl("Biometric enrollment complete!",
                           size=16, bold=True))
        lay.addSpacing(4)
        lay.addWidget(_lbl(
            "Eyecon will now verify your identity every 5 seconds during use.\n"
            "If an impostor is detected, all controls are paused automatically.",
            size=10, color=C["sec"]
        ))

        self._summary = _lbl("", size=9, color=C["ter"])
        lay.addWidget(self._summary)
        lay.addStretch()

    def set_summary(self, face_ok, hand_ok, voice_ok):
        lines = [
            f"{'✓' if face_ok  else '–'}  Face embedding   {'enrolled' if face_ok  else 'skipped'}",
            f"{'✓' if hand_ok  else '–'}  Hand signature   {'enrolled' if hand_ok  else 'skipped'}",
            f"{'✓' if voice_ok else '–'}  Voice print      {'enrolled' if voice_ok else 'skipped (no audio)'}",
        ]
        self._summary.setText("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ENROLLMENT WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class BiometricEnrollmentWindow(QMainWindow):
    """
    Show this window after register_user() succeeds.
    Connect to .enrolled(user_id) to get notified when done.
    """
    enrolled = pyqtSignal(int)   # user_id

    def __init__(self, user_id: int, username: str):
        super().__init__()
        self.user_id  = user_id
        self.username = username
        self.setWindowTitle(f"Eyecon — Biometric Setup  ({username})")
        self.setFixedSize(580, 580)
        self._center()

        self._face_emb  = None
        self._face_rat  = None
        self._hand_sig  = None
        self._voice_mfcc = None

        self._signals  = EnrollSignals()
        self._signals.progress.connect(self._on_progress)
        self._signals.step_done.connect(self._on_step_done)
        self._signals.all_done.connect(self._on_all_done)
        self._signals.error.connect(self._on_error)

        self._build_ui()
        self._cap = None

    def _center(self):
        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - 580) // 2,
                  (screen.height() - 580) // 2)

    # ─── UI ──────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QWidget()
        root.setStyleSheet(f"background:{C['bg']};")
        self.setCentralWidget(root)
        main_lay = QVBoxLayout(root)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)

        # Header
        hdr = QWidget()
        hdr.setFixedHeight(50)
        hdr.setStyleSheet(
            f"background:{C['panel']}; border-bottom:1px solid {C['border']};"
        )
        h_lay = QHBoxLayout(hdr)
        h_lay.setContentsMargins(16, 0, 16, 0)
        logo = QLabel()
        logo.setTextFormat(Qt.TextFormat.RichText)
        logo.setText(
            f'<span style="font-family:Consolas;font-size:14px;font-weight:bold;'
            f'color:{C["accent"]}">EYE</span>'
            f'<span style="font-family:Consolas;font-size:14px;font-weight:bold;'
            f'color:{C["text"]}">CON</span>'
            f'<span style="font-family:Consolas;font-size:10px;color:{C["ter"]}"> — Biometric Setup</span>'
        )
        h_lay.addWidget(logo)
        h_lay.addStretch()
        self._step_indicator = QLabel("Step 1 of 4")
        self._step_indicator.setFont(QFont("Consolas", 9))
        self._step_indicator.setStyleSheet(f"color:{C['ter']}; background:transparent;")
        h_lay.addWidget(self._step_indicator)
        main_lay.addWidget(hdr)

        # Step progress dots
        dots_w = QWidget()
        dots_w.setFixedHeight(28)
        dots_w.setStyleSheet("background:transparent;")
        dots_lay = QHBoxLayout(dots_w)
        dots_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dots_lay.setSpacing(8)
        self._dots = []
        for i in range(5):
            d = QLabel()
            d.setFixedSize(8, 8)
            d.setStyleSheet(f"background:{C['ter']}; border-radius:4px;")
            dots_lay.addWidget(d)
            self._dots.append(d)
        main_lay.addWidget(dots_w)

        # Stacked pages
        self._stack = QStackedWidget()
        self._stack.setStyleSheet("background:transparent;")
        main_lay.addWidget(self._stack, 1)

        self._welcome_page = WelcomePage()
        self._face_page    = CapturePage(
            "Face capture",
            "Look directly at the camera. Keep your face well-lit and stay still."
        )
        self._hand_page    = CapturePage(
            "Hand capture",
            "Show each hand pose clearly in front of the camera."
        )
        self._voice_page   = VoicePage()
        self._done_page    = DonePage()

        for p in [self._welcome_page, self._face_page,
                  self._hand_page, self._voice_page, self._done_page]:
            self._stack.addWidget(p)

        # Footer buttons
        footer = QWidget()
        footer.setFixedHeight(60)
        footer.setStyleSheet(
            f"background:{C['panel']}; border-top:1px solid {C['border']};"
        )
        f_lay = QHBoxLayout(footer)
        f_lay.setContentsMargins(16, 0, 16, 0)

        self._skip_btn = QPushButton("Skip biometrics")
        self._skip_btn.setFont(QFont("Consolas", 10))
        self._skip_btn.setFixedHeight(36)
        self._skip_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._skip_btn.setStyleSheet(
            f"QPushButton {{background:transparent; color:{C['ter']};"
            f" border:none; font-family:Consolas; font-size:10px;}}"
            f"QPushButton:hover {{color:{C['sec']};}}"
        )
        self._skip_btn.clicked.connect(self._skip)

        self._next_btn = QPushButton("Start  →")
        self._next_btn.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        self._next_btn.setFixedHeight(36)
        self._next_btn.setMinimumWidth(120)
        self._next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._next_btn.setStyleSheet(
            f"QPushButton {{background:{C['accent']}; color:#0d1117;"
            f" border:none; border-radius:6px; padding:0 16px;"
            f" font-family:Consolas; font-weight:bold;}}"
            f"QPushButton:hover {{background:#79b8ff;}}"
            f"QPushButton:disabled {{background:{C['border']}; color:{C['ter']};}}"
        )
        self._next_btn.clicked.connect(self._next_step)

        f_lay.addWidget(self._skip_btn)
        f_lay.addStretch()
        f_lay.addWidget(self._next_btn)
        main_lay.addWidget(footer)

        self._current_step = 0
        self._update_dots()

    # ─── NAVIGATION ──────────────────────────────────────────────────
    def _update_dots(self):
        for i, d in enumerate(self._dots):
            if i == self._current_step:
                d.setStyleSheet(
                    f"background:{C['accent']}; border-radius:4px;"
                )
            elif i < self._current_step:
                d.setStyleSheet(
                    f"background:{C['green']}; border-radius:4px;"
                )
            else:
                d.setStyleSheet(
                    f"background:{C['ter']}; border-radius:4px;"
                )
        self._step_indicator.setText(
            f"Step {self._current_step + 1} of {len(self._dots)}"
        )

    def _next_step(self):
        if self._current_step == 0:
            # Welcome → Face
            self._current_step = 1
            self._stack.setCurrentIndex(1)
            self._next_btn.setEnabled(False)
            self._next_btn.setText("Capturing…")
            self._update_dots()
            self._start_face_capture()

        elif self._current_step == 2:
            # Hand page shown, next → Voice
            self._current_step = 3
            self._stack.setCurrentIndex(3)
            self._next_btn.setEnabled(False)
            self._update_dots()
            self._start_voice_capture()

        elif self._current_step == 4:
            # Done → launch
            self._finish()

    def _on_step_done(self, step):
        if step == "face":
            self._face_page.set_done()
            # Auto-advance to hand after 0.8s
            QTimer.singleShot(800, self._start_hand_capture)

        elif step == "hand":
            self._hand_page.set_done()
            self._current_step = 2
            self._next_btn.setEnabled(True)
            self._next_btn.setText("Next  →")
            self._update_dots()

        elif step == "voice":
            self._voice_page.set_done()
            QTimer.singleShot(600, self._show_done)

    def _show_done(self):
        self._current_step = 4
        self._stack.setCurrentIndex(4)
        self._next_btn.setEnabled(True)
        self._next_btn.setText("Launch Eyecon  →")
        self._skip_btn.hide()
        self._update_dots()

        face_ok  = self._face_emb  is not None
        hand_ok  = self._hand_sig  is not None
        voice_ok = self._voice_mfcc is not None

        self._done_page.set_summary(face_ok, hand_ok, voice_ok)

        # Save to CSV
        threading.Thread(target=self._save, daemon=True).start()

    def _save(self):
        from modules.biometric_enroller import save_biometrics
        save_biometrics(
            self.user_id,
            self._face_emb,
            self._face_rat,
            self._hand_sig,
            self._voice_mfcc,
        )

    def _finish(self):
        if self._cap:
            self._cap.release()
        self.enrolled.emit(self.user_id)
        self.close()

    def _skip(self):
        from modules.biometric_enroller import save_biometrics
        save_biometrics(self.user_id, None, None, None, None)
        if self._cap:
            self._cap.release()
        self.enrolled.emit(self.user_id)
        self.close()

    # ─── CAPTURE THREADS ─────────────────────────────────────────────
    def _open_cap(self):
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(0)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def _start_face_capture(self):
        self._stack.setCurrentIndex(1)
        self._current_step = 1
        self._update_dots()

        def run():
            self._open_cap()
            import mediapipe as mp
            fm = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.6)

            def cb(step, done, total, frame):
                self._signals.progress.emit(step, done, total, frame, "")

            from modules.biometric_enroller import capture_face_embedding
            emb, rat = capture_face_embedding(self._cap, fm, 30, cb)
            fm.close()
            self._face_emb = emb
            self._face_rat = rat
            self._signals.step_done.emit("face")

        threading.Thread(target=run, daemon=True).start()

    def _start_hand_capture(self):
        self._current_step = 2
        self._stack.setCurrentIndex(2)
        self._next_btn.setEnabled(False)
        self._next_btn.setText("Capturing…")
        self._update_dots()

        def run():
            self._open_cap()
            import mediapipe as mp
            hd = mp.solutions.hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.7)

            def cb(step, done, total, frame, pose_name=""):
                self._signals.progress.emit(step, done, total, frame, pose_name)

            from modules.biometric_enroller import capture_hand_signature
            sig = capture_hand_signature(self._cap, hd, cb)
            hd.close()
            self._hand_sig = sig
            self._signals.step_done.emit("hand")

        threading.Thread(target=run, daemon=True).start()

    def _start_voice_capture(self):
        from modules.biometric_enroller import _VOICE_PHRASES

        def run():
            def cb(event, phrase_idx, total, phrase):
                if event == "voice_ready":
                    self._signals.progress.emit(
                        "voice_ready", phrase_idx, total, None, phrase)
                elif event == "voice_recording":
                    self._signals.progress.emit(
                        "voice_recording", phrase_idx, total, None, phrase)
                elif event == "voice_done":
                    self._signals.progress.emit(
                        "voice_done", phrase_idx, total, None, phrase)

            from modules.biometric_enroller import capture_voice_mfcc
            mfcc = capture_voice_mfcc(cb)
            self._voice_mfcc = mfcc
            self._signals.step_done.emit("voice")

        threading.Thread(target=run, daemon=True).start()

    # ─── SIGNAL HANDLERS ─────────────────────────────────────────────
    def _on_progress(self, step, done, total, frame, extra):
        if step == "face":
            self._face_page.update_progress(done, total, frame)
        elif step == "hand":
            self._hand_page.update_progress(done, total, frame, extra)
        elif step == "voice_ready":
            self._voice_page.set_phrase(extra, done, total)
            self._voice_page.set_status(f"Get ready to say this phrase…",
                                        C["sec"])
        elif step == "voice_recording":
            self._voice_page.set_phrase(extra, done, total)
            self._voice_page.set_status("🔴  Recording…", C["red"])
        elif step == "voice_done":
            self._voice_page.set_status("✓  Phrase captured", C["green"])

    def _on_all_done(self, user_id):
        self._show_done()

    def _on_error(self, msg):
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.warning(self, "Enrollment error", msg)

    def closeEvent(self, event):
        if self._cap:
            self._cap.release()
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = BiometricEnrollmentWindow(user_id=1, username="testuser")
    win.enrolled.connect(lambda uid: print(f"Enrolled user {uid}"))
    win.show()
    sys.exit(app.exec())
