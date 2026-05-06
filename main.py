r"""
 _____ _  _ ___ ___  ___  _  _
| __| || | __/ __|/ _ \| \| |
| _| \_, | _|(__| (_) | .` |
|___| |_/|___\___|\\___/|_|\\_|

Eyecon — AI-Powered Multimodal Interaction System
Author: Eyecon Team
Version: 2.1.0
"""

import argparse
import cv2
import threading
import time
import signal
import sys
import os
from modules.eye_tracker     import EyeTracker
from modules.gesture_engine  import GestureEngine
from modules.voice_engine    import VoiceEngine
from modules.ai_decision     import AIDecisionModule
from modules.feedback        import FeedbackSystem
from utils.logger            import EyeconLogger
from utils.config            import Config

# ─── Graceful shutdown ────────────────────────────────────────────────────────
running = True
def _shutdown(sig, frame):
    global running
    print("\n[Eyecon] Shutdown signal received. Stopping…")
    running = False

signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


def run_cli():
    logger = EyeconLogger("Main")
    cfg    = Config("config/settings.json")

    logger.info("=" * 50)
    logger.info("  EYECON Multimodal System  v2.1.0")
    logger.info("=" * 50)

    # ── Init shared webcam capture ─────────────────────────────────────────
    cap = cv2.VideoCapture(cfg.get("camera_index", 0))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.get("cam_width",  640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.get("cam_height", 480))
    cap.set(cv2.CAP_PROP_FPS,          cfg.get("cam_fps",     30))

    if not cap.isOpened():
        logger.error("Cannot open webcam. Check camera_index in config.")
        sys.exit(1)

    # ── Init modules ───────────────────────────────────────────────────────
    feedback  = FeedbackSystem()
    eye       = EyeTracker(cfg, feedback)
    gesture   = GestureEngine(cfg, feedback)
    voice     = VoiceEngine(cfg, feedback)
    ai_module = AIDecisionModule(eye, gesture, voice, feedback, cfg)

    # ── Start voice listener in background thread ──────────────────────────
    voice_thread = threading.Thread(target=voice.listen_loop, daemon=True)
    voice_thread.start()
    logger.info("Voice engine started (background thread)")

    # ── Calibrate eye tracker ──────────────────────────────────────────────
    if cfg.get("auto_calibrate", True):
        logger.info("Starting eye calibration… follow the dots on screen.")
        eye.calibrate(cap)

    logger.info("All systems online. Press 'Q' in the window or say 'Stop Eyecon' to quit.")

    frame_count = 0
    fps_timer   = time.time()

    # ═══════════════════════════════════════════════════════════════════════
    #  MAIN LOOP
    # ═══════════════════════════════════════════════════════════════════════
    while running and not voice.stop_requested:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Frame capture failed — retrying…")
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)          # mirror for natural feel

        # ── Run each module on the current frame ───────────────────────
        eye_data     = eye.process(frame)
        gesture_data = gesture.process(frame)

        # ── AI Decision Layer decides what action to take ──────────────
        action = ai_module.decide(eye_data, gesture_data, frame=frame)
        if action:
            ai_module.execute(action)

        # ── Draw debug overlays ────────────────────────────────────────
        eye.draw_overlay(frame, eye_data)
        gesture.draw_overlay(frame, gesture_data)
        ai_module.draw_status(frame, action)

        # ── FPS counter ────────────────────────────────────────────────
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_timer
            fps     = 30 / elapsed if elapsed > 0 else 0
            fps_timer = time.time()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 1)

        cv2.imshow("Eyecon — Multimodal Interface", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    # ── Cleanup ────────────────────────────────────────────────────────────
    logger.info("Shutting down Eyecon…")
    cap.release()
    cv2.destroyAllWindows()
    voice.stop()
    eye.cleanup()
    gesture.cleanup()
    feedback.speak("Eyecon stopped. Goodbye.")
    logger.info("All systems offline. Session saved.")


def run_gui():
    """Launch the PyQt6 native GUI with authentication."""
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
        import auth_window
        import app as eyecon_app
    except ImportError as exc:
        print(f"[Eyecon] GUI dependencies not installed ({exc}). Run: pip install PyQt6")
        return False

    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.window.warning=false")
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    qapp = QApplication.instance() or QApplication(sys.argv)
    qapp.setApplicationName("Eyecon")
    qapp.setApplicationVersion("2.1.0")
    c = eyecon_app.C
    qapp.setStyleSheet(
        f"""
        QToolTip {{ background: {c['panel']}; color: {c['text_primary']};
                   border: 1px solid {c['border']}; font-family: Consolas; font-size: 10px; }}
        QScrollBar:vertical {{ background: transparent; width: 6px; }}
        QScrollBar::handle:vertical {{ background: {c['border_hi']}; border-radius: 3px; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
        """
    )

    auth = auth_window.AuthWindow()
    win_ref = {"win": None}

    def _open_app(user):
        win_ref["win"] = eyecon_app.EyeconWindow()
        win_ref["win"].show()
        # Tell AI module who logged in so it loads their biometric profile
        if eyecon_app.BACKEND_AVAILABLE and hasattr(win_ref["win"], '_worker'):
            worker = win_ref["win"]._worker
            if hasattr(worker, '_ai'):
                worker._ai.attach_verifier(user["id"])
            if hasattr(worker, '_cfg'):
                worker._cfg.set("bio_user_id", user["id"])

    auth.authenticated.connect(_open_app)
    auth.show()
    qapp.exec()
    return True


def main():
    parser = argparse.ArgumentParser(description="Eyecon launcher")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gui", action="store_true", help="Launch the PyQt UI")
    mode.add_argument("--cli", action="store_true", help="Run the CLI pipeline")
    args = parser.parse_args()

    if args.cli:
        run_cli()
        return

    # Default: try GUI first, fall back to CLI
    if run_gui():
        return

    if args.gui:
        sys.exit(1)

    run_cli()


if __name__ == "__main__":
    main()
