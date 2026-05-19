"""
modules/ai_decision.py  —  v2.0  STATE MACHINE FIXED
──────────────────────────────────────────────────────
All bugs from v1 fixed:
  FIX 1 — RESUME re-enables eye tracker (eye.enabled=True, system_paused=False)
  FIX 2 — PAUSE properly disables eye cursor movement via system_paused flag
  FIX 3 — System starts PAUSED, OPEN_PALM resumes (matching gesture_engine)
  FIX 4 — Gesture always processed even when paused (to detect OPEN_PALM)
  FIX 5 — Voice/FRIDAY priority correctly suppresses gesture for 30 frames

PLACE AT: ML-Model/modules/ai_decision.py  (replace existing)
"""

import cv2
import time
import collections
from utils.logger import EyeconLogger

STATE_ACTIVE = "ACTIVE"
STATE_PAUSED = "PAUSED"


class AIDecisionModule:

    def __init__(self, eye, gesture, voice, feedback, config):
        self.eye      = eye
        self.gesture  = gesture
        self.voice    = voice      # now FridayEngine
        self.feedback = feedback
        self.cfg      = config
        self.logger   = EyeconLogger("AIDecision")

        # Start PAUSED — OPEN_PALM will resume
        self.state = STATE_PAUSED
        self._apply_state(STATE_PAUSED)

        self.last_action = None
        self.last_source = None

        # Noise filters
        self._eye_buf   = collections.deque(maxlen=5)
        self._gest_buf  = collections.deque(maxlen=5)
        self._voice_buf = collections.deque(maxlen=3)

        # Event log
        self._events = collections.deque(maxlen=100)

        # Timing
        self._last_voice_t   = 0.0
        self._voice_window   = config.get("voice_speaking_secs",  2.0)
        self._action_cd      = 0

        # Optional biometric verifier
        self._bio = None

        self.decision_count = 0
        self.logger.info("AIDecisionModule v2.0 — starts PAUSED, open palm to activate")

    # ─────────────────────────────────────────────────────────────────────────
    #  STATE
    # ─────────────────────────────────────────────────────────────────────────
    def _apply_state(self, new_state: str):
        """Single place that enables/disables all subsystems."""
        self.state = new_state
        if new_state == STATE_PAUSED:
            self.eye.enabled          = False
            self.eye.system_paused    = True
            self.gesture.paused       = True
            # gesture.enabled stays True so OPEN_PALM is still detected
        elif new_state == STATE_ACTIVE:
            self.eye.enabled          = True
            self.eye.system_paused    = False
            self.gesture.paused       = False
            self.gesture.enabled      = True
        self.logger.info(f"System state -> {new_state}")

    # ─────────────────────────────────────────────────────────────────────────
    #  MAIN DECIDE  — called every frame
    # ─────────────────────────────────────────────────────────────────────────
    def decide(self, eye_data, gesture_data, frame=None):
        now = time.time()
        self._action_cd = max(0, self._action_cd - 1)

        # Biometric check
        if self._bio is not None and frame is not None:
            bio = self._bio.process(frame)
            if bio.get("impostor"):
                return None

        # Poll FRIDAY for last command
        voice_cmd = None
        if hasattr(self.voice, "get_last_command"):
            voice_cmd = self.voice.get_last_command()
        if voice_cmd:
            self._last_voice_t = now
            self._voice_buf.append(voice_cmd)

        voice_recent = (now - self._last_voice_t) < self._voice_window

        # Build candidates
        candidates = []

        # Voice — highest priority
        if voice_cmd:
            candidates.append({
                "source": "VOICE", "priority": 3,
                "action": voice_cmd, "data": {},
            })

        # Gesture — always checked (even paused, for OPEN_PALM/PAUSE)
        g_action   = gesture_data.get("action")   if gesture_data.get("active") else None
        g_executed = gesture_data.get("executed")  if gesture_data.get("active") else False

        if g_executed and g_action and not voice_recent:
            candidates.append({
                "source": "GESTURE", "priority": 2,
                "action": g_action, "data": gesture_data,
            })

        # Eye — only when active and system not paused
        if self.state == STATE_ACTIVE and not voice_recent:
            e_blink = eye_data.get("blink")       if eye_data.get("active") else None
            e_dwell = eye_data.get("dwell_click")  if eye_data.get("active") else None
            if e_blink or e_dwell:
                candidates.append({
                    "source": "EYE", "priority": 1,
                    "action": "BLINK_CLICK" if e_blink else "DWELL_CLICK",
                    "data":   eye_data,
                })

        if not candidates:
            return None

        # Pick highest priority
        winner = max(candidates, key=lambda c: c["priority"])

        # State-change actions bypass all gates
        is_state_change = winner["action"] in ("RESUME", "PAUSE")

        # Noise gate
        buf = {"VOICE": self._voice_buf,
               "GESTURE": self._gest_buf,
               "EYE": self._eye_buf}.get(winner["source"], self._eye_buf)
        if not is_state_change and list(buf).count(winner["action"]) >= 4:
            return None

        # Cooldown gate
        if (self._action_cd > 0
                and winner["source"] != "VOICE"
                and not is_state_change):
            return None

        # Commit
        buf.append(winner["action"])
        self.last_action = winner["action"]
        self.last_source = winner["source"]
        self._action_cd  = self.cfg.get("ai_action_cooldown_frames", 12)
        self.decision_count += 1
        self._events.append({
            "t": now, "source": winner["source"], "action": winner["action"]
        })

        # Suppress gesture briefly if voice fired
        if winner["source"] == "VOICE":
            self.gesture._action_cd = max(
                getattr(self.gesture, "_action_cd", 0), 30)

        return winner

    # ─────────────────────────────────────────────────────────────────────────
    #  EXECUTE
    # ─────────────────────────────────────────────────────────────────────────
    def execute(self, action_dict):
        action = action_dict.get("action", "")

        # ── RESUME ────────────────────────────────────────────────────
        if action == "RESUME":
            if self.state != STATE_ACTIVE:
                self._apply_state(STATE_ACTIVE)
                self.feedback.speak("System active")
                self.logger.info("RESUMED — eye tracking enabled")

        # ── PAUSE ─────────────────────────────────────────────────────
        elif action == "PAUSE":
            if self.state != STATE_PAUSED:
                self._apply_state(STATE_PAUSED)
                self.feedback.speak("System paused")
                self.logger.info("PAUSED — eye tracking disabled")

    # ─────────────────────────────────────────────────────────────────────────
    #  BIOMETRIC
    # ─────────────────────────────────────────────────────────────────────────
    def attach_verifier(self, user_id: int):
        try:
            from modules.biometric_verifier import BiometricVerifier
            self._bio = BiometricVerifier(user_id, self.cfg)
            self._bio.on_impostor = self._on_impostor
            self._bio.on_resumed  = self._on_bio_clear
            self.logger.info(f"Biometric verifier attached for user {user_id}")
        except Exception as e:
            self.logger.warning(f"Biometric verifier unavailable: {e}")

    def _on_impostor(self, score):
        self._apply_state(STATE_PAUSED)
        self.feedback.speak("Unrecognised user. System paused.")
        self.logger.warning(f"IMPOSTOR (score={score:.3f})")

    def _on_bio_clear(self):
        self._apply_state(STATE_ACTIVE)
        self.feedback.speak("Identity confirmed. Resuming.")

    # ─────────────────────────────────────────────────────────────────────────
    #  OVERLAY
    # ─────────────────────────────────────────────────────────────────────────
    def draw_status(self, frame, action):
        h, w = frame.shape[:2]
        col  = (0, 180, 80) if self.state == STATE_ACTIVE else (50, 50, 160)
        cv2.putText(frame,
                    f"[AI] {self.state}  V>G>E  #{self.decision_count}",
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1)
        if action:
            cv2.putText(frame,
                        f"  {action['source']} -> {action['action']}",
                        (10, h - 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 180, 0), 1)
        if self._bio:
            self._bio.draw_overlay(frame)

    # ─────────────────────────────────────────────────────────────────────────
    def get_stats(self):
        counts = {"VOICE": 0, "GESTURE": 0, "EYE": 0}
        for e in self._events:
            counts[e["source"]] = counts.get(e["source"], 0) + 1
        return {"total": self.decision_count,
                "breakdown": counts,
                "last_action": self.last_action,
                "state": self.state}