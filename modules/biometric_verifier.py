"""
modules/biometric_verifier.py
──────────────────────────────
Runtime Biometric Verification

Called from ai_decision.py every 5 seconds while the app is running.

Checks:
  FACE  — cosine similarity between live face ratios and stored ones
          Threshold: 0.80  (tunable in config)
          Uses the 20-float ratio vector (fast, not the full 1404 embedding)

  VOICE — compared only when a voice command is detected
          MFCC cosine similarity, threshold 0.75

Logic:
  • 3 consecutive face failures → impostor flag raised
  • ai_decision.py checks the flag each frame
  • On flag: pause ALL input, show overlay alert, log security event
  • User can re-authenticate by looking at camera for 3s matching

Grace period: 15s — gives time for lighting changes, angle shifts
"""

import numpy as np
import time
import threading
import os
import cv2

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from modules.biometric_enroller import (
    _extract_face_embedding,
    _compute_mfcc,
    load_biometrics,
    _AUDIO_OK,
)
from utils.logger import EyeconLogger

logger = EyeconLogger("Verifier")


def _cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class BiometricVerifier:
    """
    Lightweight runtime verifier.
    Instantiate once after login; call .process(frame) each verification tick.
    """

    def __init__(self, user_id: int, config):
        self.user_id = user_id
        self.cfg     = config
        self._stored = load_biometrics(user_id)

        if self._stored is None:
            logger.warning(f"No biometrics found for user {user_id} — verification disabled")
            self.enabled = False
        else:
            self.enabled = True
            logger.info(f"BiometricVerifier ready for user {user_id} "
                        f"(enrolled {self._stored['enrolled_at']})")

        # ── MediaPipe Tasks API: FaceLandmarker (lightweight) ─────────
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "face_landmarker.task"
        )
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model_data = f.read()
            base_options = mp_python.BaseOptions(model_asset_buffer=model_data)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.4,
                min_tracking_confidence=0.4,
            )
            self._face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        else:
            self._face_landmarker = None
            logger.warning(f"FaceLandmarker model not found at {model_path} — bio face check disabled")

        # ── State ─────────────────────────────────────────────────────
        self._face_fail_streak = 0
        self._face_pass_streak = 0
        self.impostor_detected  = False
        self.last_score         = 1.0
        self.last_check_t       = 0.0
        self._grace_until       = 0.0   # ignore failures until this time

        # Config
        self.check_interval_s    = config.get("bio_check_interval_secs",  5)
        self.face_threshold      = config.get("bio_face_threshold",        0.80)
        self.voice_threshold     = config.get("bio_voice_threshold",       0.75)
        self.fail_streak_limit   = config.get("bio_fail_streak_limit",     3)
        self.pass_streak_reauth  = config.get("bio_pass_streak_reauth",    3)
        self.grace_period_s      = config.get("bio_grace_period_secs",    15)

        # Alert callback — set by ai_decision
        self.on_impostor: callable = None
        self.on_resumed:  callable = None

        # Security log
        self._events = []

        self._lock = threading.Lock()

    # ─────────────────────────────────────────────────────────────────
    #  MAIN ENTRY — called every frame from ai_decision, but only does
    #  work every check_interval_s seconds
    # ─────────────────────────────────────────────────────────────────
    def process(self, frame) -> dict:
        """
        Args:  frame  — current BGR webcam frame
        Returns: result dict with keys:
            verified (bool), score (float), impostor (bool), skipped (bool)
        """
        if not self.enabled:
            return {"verified": True, "score": 1.0,
                    "impostor": False, "skipped": True}

        now = time.time()

        # Only run every check_interval_s
        if now - self.last_check_t < self.check_interval_s:
            return {"verified": not self.impostor_detected,
                    "score": self.last_score,
                    "impostor": self.impostor_detected,
                    "skipped": True}

        self.last_check_t = now

        # ── Face check ────────────────────────────────────────────────
        score, verified = self._verify_face(frame)
        self.last_score  = score

        with self._lock:
            if verified:
                self._face_fail_streak  = 0
                self._face_pass_streak += 1

                # Re-auth after impostor flag if user looks at camera
                if (self.impostor_detected and
                        self._face_pass_streak >= self.pass_streak_reauth):
                    self.impostor_detected = False
                    self._face_pass_streak = 0
                    self._log_event("RESUMED", score)
                    logger.info(f"User {self.user_id} re-authenticated (score={score:.3f})")
                    if self.on_resumed:
                        self.on_resumed()

            else:
                self._face_pass_streak = 0

                # Respect grace period (lighting change, quick look away)
                if now < self._grace_until:
                    logger.debug(f"Face fail during grace period (score={score:.3f})")
                else:
                    self._face_fail_streak += 1
                    logger.warning(f"Face verification failed "
                                   f"(score={score:.3f}, streak={self._face_fail_streak})")

                    if (not self.impostor_detected and
                            self._face_fail_streak >= self.fail_streak_limit):
                        self.impostor_detected = True
                        self._log_event("IMPOSTOR", score)
                        logger.warning(
                            f"IMPOSTOR DETECTED for user {self.user_id}! "
                            f"score={score:.3f}"
                        )
                        if self.on_impostor:
                            self.on_impostor(score)

        return {
            "verified": verified,
            "score":    score,
            "impostor": self.impostor_detected,
            "skipped":  False,
        }

    # ─────────────────────────────────────────────────────────────────
    #  FACE VERIFY
    # ─────────────────────────────────────────────────────────────────
    def _verify_face(self, frame):
        if self._face_landmarker is None:
            return 1.0, True   # model not loaded → pass

        h, w = frame.shape[:2]
        # Work on a copy to avoid mutating the display frame
        rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._face_landmarker.detect(mp_image)

        if not result.face_landmarks:
            # No face detected — don't count as failure (user may have looked away)
            # Trigger grace period
            self._grace_until = time.time() + self.grace_period_s
            return self.last_score, True   # hold last score

        lm = result.face_landmarks[0]
        _, live_ratios = _extract_face_embedding(lm, w, h)

        stored_ratios = self._stored.get("face_ratios")
        if stored_ratios is None:
            return 1.0, True   # no stored data → pass

        score = _cosine_similarity(live_ratios, stored_ratios)
        return score, score >= self.face_threshold

    # ─────────────────────────────────────────────────────────────────
    #  VOICE VERIFY  (called ad-hoc when voice command fires)
    # ─────────────────────────────────────────────────────────────────
    def verify_voice(self, audio_array) -> tuple:
        """
        Returns (score, verified) for a captured audio clip.
        Called from voice_engine when a command fires.
        """
        if not self.enabled or not self._stored.get("voice_enabled"):
            return 1.0, True

        stored_mfcc = self._stored.get("voice_mfcc")
        if stored_mfcc is None:
            return 1.0, True

        live_mfcc = _compute_mfcc(audio_array)
        score     = _cosine_similarity(live_mfcc, stored_mfcc)
        verified  = score >= self.voice_threshold

        if not verified:
            logger.warning(f"Voice verification failed (score={score:.3f})")

        return score, verified

    # ─────────────────────────────────────────────────────────────────
    #  LOGIN-TIME FACE CHECK  (one-off, called right after password OK)
    # ─────────────────────────────────────────────────────────────────
    def verify_login_face(self, cap, n_frames=15) -> tuple:
        """
        Quick face check at login — takes n_frames from cap.
        Returns (mean_score, verified).
        Called by auth_window after successful password login.
        """
        if not self.enabled or self._face_landmarker is None:
            return 1.0, True

        stored_ratios = self._stored.get("face_ratios")
        if stored_ratios is None:
            return 1.0, True

        scores = []
        captured = 0

        while captured < n_frames:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            frame  = cv2.flip(frame, 1)
            h, w   = frame.shape[:2]
            rgb    = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._face_landmarker.detect(mp_image)

            if result.face_landmarks:
                lm = result.face_landmarks[0]
                _, ratios = _extract_face_embedding(lm, w, h)
                scores.append(_cosine_similarity(ratios, stored_ratios))
                captured += 1

        if not scores:
            logger.warning("No face detected during login check")
            return 0.0, False

        mean_score = float(np.mean(scores))
        verified   = mean_score >= self.face_threshold
        self._log_event("LOGIN_FACE_PASS" if verified else "LOGIN_FACE_FAIL",
                        mean_score)
        logger.info(f"Login face check: score={mean_score:.3f} → {'PASS' if verified else 'FAIL'}")
        return mean_score, verified

    # ─────────────────────────────────────────────────────────────────
    #  OVERLAY DRAWING  (called by ai_decision.py)
    # ─────────────────────────────────────────────────────────────────
    def draw_overlay(self, frame):
        h, w = frame.shape[:2]
        score = self.last_score

        if self.impostor_detected:
            # Red pulsing border
            cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 220), 4)
            cv2.putText(frame,
                        "⚠  UNRECOGNISED USER — SYSTEM PAUSED",
                        (10, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 255), 2)
            cv2.putText(frame,
                        "Look at camera to re-authenticate",
                        (10, h - 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 120, 255), 1)
        else:
            color = (0, 220, 80) if score >= self.face_threshold else (0, 160, 255)
            cv2.putText(frame,
                        f"[Bio] face={score:.2f}",
                        (w - 160, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        color, 1)

    # ─────────────────────────────────────────────────────────────────
    #  LOGGING
    # ─────────────────────────────────────────────────────────────────
    def _log_event(self, event_type, score):
        self._events.append({
            "t":     time.strftime("%Y-%m-%dT%H:%M:%S"),
            "type":  event_type,
            "score": round(score, 4),
        })
        # Write to security log in data/
        log_path = os.path.join(
            os.path.dirname(__file__), "..", "data",
            f"security_user{self.user_id}.csv"
        )
        import csv, os
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        write_header = not os.path.exists(log_path)
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["t", "type", "score"])
            if write_header:
                w.writeheader()
            w.writerow(self._events[-1])

    def cleanup(self):
        if self._face_landmarker is not None:
            self._face_landmarker.close()
        logger.info(f"BiometricVerifier closed. Events logged: {len(self._events)}")


import os   # needed for _log_event
