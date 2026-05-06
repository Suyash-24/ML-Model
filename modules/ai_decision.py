"""
modules/ai_decision.py
──────────────────────
AI Decision & Control Module  —  The Brain of Eyecon

Responsibilities:
  • Resolve conflicts when multiple inputs fire simultaneously
  • Apply priority logic: VOICE > GESTURE > EYE
  • Context-awareness: pause gesture detection while user speaks
  • Filter accidental inputs (noise, micro-movements, phantom gestures)
  • Maintain interaction state machine
  • Log decisions for debugging / ML improvement
"""

import cv2
import time
import collections
from utils.logger import EyeconLogger
from modules.biometric_verifier import BiometricVerifier


# ── Priority levels (higher = wins when conflict) ────────────────────────────
PRIORITY = {"VOICE": 3, "GESTURE": 2, "EYE": 1}

# ── States ───────────────────────────────────────────────────────────────────
STATE_ACTIVE  = "ACTIVE"
STATE_PAUSED  = "PAUSED"
STATE_VOICE   = "VOICE_PRIORITY"
STATE_GESTURE = "GESTURE_PRIORITY"
STATE_EYE     = "EYE_PRIORITY"


class AIDecisionModule:
    def __init__(self, eye, gesture, voice, feedback, config):
        self.eye      = eye
        self.gesture  = gesture
        self.voice    = voice
        self.feedback = feedback
        self.cfg      = config
        self.logger   = EyeconLogger("AIDecision")

        # Interaction state
        start_paused = self.cfg.get("start_paused", True)
        self.state        = STATE_PAUSED if start_paused else STATE_ACTIVE
        self.active_mode  = "GESTURE" if start_paused else "MULTI"   # MULTI | EYE | GESTURE | VOICE
        self.last_action  = None
        self.last_source  = None
        self.last_action_t = 0

        # Noise filter: ring buffers for each modality
        self._eye_actions     = collections.deque(maxlen=5)
        self._gesture_actions = collections.deque(maxlen=5)
        self._voice_actions   = collections.deque(maxlen=3)

        # Context window (recent events)
        self._event_log   = collections.deque(maxlen=50)

        # Silence / inactivity tracking
        self._last_eye_active     = time.time()
        self._last_gesture_active = time.time()
        self._last_voice_active   = time.time()
        self._eye_inactive_thresh     = config.get("eye_inactive_secs",     10)
        self._gesture_inactive_thresh = config.get("gesture_inactive_secs", 15)
        self._voice_speaking_window   = config.get("voice_speaking_secs",    2)

        # Cooldowns
        self._action_cooldown = 0
        self.decision_count   = 0

        # ── Biometric verifier ─────────────────────────────────────────────
        self._bio: BiometricVerifier | None = None
        if hasattr(config, '_user_id') or config.get('bio_user_id'):
            uid = config.get('bio_user_id')
            if uid:
                self._bio = BiometricVerifier(uid, config)
                self._bio.on_impostor = self._on_impostor_detected
                self._bio.on_resumed  = self._on_impostor_cleared

        self.logger.info("AI Decision Module online — priority: VOICE > GESTURE > EYE")
        if start_paused:
            self.eye.enabled = False
            self.gesture.enabled = True
            self.gesture.actions_enabled = False
            self.logger.info("System starts PAUSED — open palm to resume")

    # ─────────────────────────────────────────────────────────────────────
    #  BIOMETRIC INTEGRATION
    # ─────────────────────────────────────────────────────────────────────
    def attach_verifier(self, user_id: int):
        """Call this right after login with the logged-in user's id."""
        self._bio = BiometricVerifier(user_id, self.cfg)
        self._bio.on_impostor = self._on_impostor_detected
        self._bio.on_resumed  = self._on_impostor_cleared

    def _on_impostor_detected(self, score: float):
        self.state            = "PAUSED"
        self.eye.enabled      = False
        self.gesture.enabled  = False
        self.feedback.speak("Unrecognised user. System paused.")
        self.logger.warning(f"IMPOSTOR — all input disabled (score={score:.3f})")

    def _on_impostor_cleared(self):
        self.state            = "ACTIVE"
        self.eye.enabled      = True
        self.gesture.enabled  = True
        self.feedback.speak("Identity confirmed. Resuming.")
        self.logger.info("Identity verified — system resumed")

    # ─────────────────────────────────────────────────────────────────────
    #  MAIN DECISION FUNCTION  (called every frame)
    # ─────────────────────────────────────────────────────────────────────
    def decide(self, eye_data, gesture_data, frame=None):
        """
        Receive current eye and gesture data.
        Voice data is async (from voice.get_last_command()).
        Return the best action dict or None.
        """
        now = time.time()
        self._last_frame = frame
        self._action_cooldown = max(0, self._action_cooldown - 1)

        # ── Poll voice command ─────────────────────────────────────────
        voice_cmd = self.voice.get_last_command()
        if voice_cmd:
            self._last_voice_active = now
            self._voice_actions.append(voice_cmd)

        # ── Track modality activity ────────────────────────────────────
        if eye_data.get("active"):
            self._last_eye_active = now
        if gesture_data.get("active"):
            self._last_gesture_active = now

        # ── Context: is user currently speaking? ───────────────────────
        voice_recent = (now - self._last_voice_active) < self._voice_speaking_window

        # ── Context: is eye tracking inactive? ────────────────────────
        eye_inactive = (now - self._last_eye_active) > self._eye_inactive_thresh

        # ── State machine ──────────────────────────────────────────────
        paused = self.voice.paused or self.state == STATE_PAUSED
        if paused:
            g_exec = gesture_data.get("executed") if gesture_data.get("active") else None
            if g_exec == "RESUME":
                return {
                    "source":   "GESTURE",
                    "priority": PRIORITY["GESTURE"],
                    "action":   "RESUME",
                    "data":     gesture_data,
                }
            if voice_cmd in ("resume eyecon", "resume"):
                return {
                    "source":   "VOICE",
                    "priority": PRIORITY["VOICE"],
                    "action":   "RESUME",
                    "data":     {},
                }
            return None

        # ── Biometric check ────────────────────────────────────────────────
        if self._bio is not None:
            bio_result = self._bio.process(frame if frame is not None else self._last_frame)
            if bio_result.get("impostor"):
                return None   # hard block — no actions while impostor flag active

        # ── Build candidate actions from each modality ─────────────────
        candidates = []

        # Voice (highest priority)
        if voice_cmd:
            candidates.append({
                "source":   "VOICE",
                "priority": PRIORITY["VOICE"],
                "action":   voice_cmd,
                "data":     {},
            })

        # Gesture (middle priority)
        g_exec = gesture_data.get("executed") if gesture_data.get("active") else None
        if g_exec and not voice_recent:
            candidates.append({
                "source":   "GESTURE",
                "priority": PRIORITY["GESTURE"],
                "action":   g_exec,
                "data":     gesture_data,
            })

        # Eye (lowest priority) — only if gesture is inactive
        e_blink = eye_data.get("blink")      if eye_data.get("active") else None
        e_dwell = eye_data.get("dwell_click") if eye_data.get("active") else None
        if (e_blink or e_dwell) and not voice_recent:
            candidates.append({
                "source":   "EYE",
                "priority": PRIORITY["EYE"],
                "action":   "BLINK_CLICK" if e_blink else "DWELL_CLICK",
                "data":     eye_data,
            })

        if not candidates:
            return None

        # ── Conflict resolution: pick highest priority ─────────────────
        winner = max(candidates, key=lambda c: c["priority"])

        # ── Noise gate: action must differ from recent repeated actions ─
        source_buf = self._get_buf(winner["source"])
        if self._is_noise(winner["action"], source_buf):
            self.logger.debug(f"Noise filtered: {winner['action']} from {winner['source']}")
            return None

        # ── Cooldown gate ──────────────────────────────────────────────
        is_scroll = str(winner["action"]).startswith("SCROLL(")
        if self._action_cooldown > 0 and winner["source"] != "VOICE" and not is_scroll:
            return None

        # ── Commit ────────────────────────────────────────────────────
        source_buf.append(winner["action"])
        self.last_action   = winner["action"]
        self.last_source   = winner["source"]
        self.last_action_t = now
        self._action_cooldown = self.cfg.get("ai_action_cooldown_frames", 15)
        self.decision_count += 1

        self._event_log.append({
            "t":      now,
            "source": winner["source"],
            "action": winner["action"],
        })

        # ── Context adjustments post-decision ─────────────────────────
        # If voice fired, briefly pause gesture to avoid double-action
        if winner["source"] == "VOICE":
            self.gesture.gesture_cooldown = max(
                self.gesture.gesture_cooldown, 30)

        # If eye went inactive → suggest gesture mode
        if eye_inactive:
            self.active_mode = "GESTURE"
        else:
            self.active_mode = "MULTI"

        self.logger.info(
            f"[Decision #{self.decision_count}] "
            f"{winner['source']} → {winner['action']}"
            f"  (conflict: {len(candidates)} candidates)"
        )

        return winner

    # ─────────────────────────────────────────────────────────────────────
    #  EXECUTE (delegated actions that aren't handled by modality modules)
    # ─────────────────────────────────────────────────────────────────────
    def execute(self, action_dict):
        """Execute any action not already handled inline."""
        action = action_dict.get("action", "")
        source = action_dict.get("source", "")

        if action == "PAUSE":
            self.state = STATE_PAUSED
            self.eye.enabled     = False
            self.gesture.enabled = True
            self.gesture.actions_enabled = False
            self.logger.info("System PAUSED by AI decision")

        elif action == "RESUME":
            self.state = STATE_ACTIVE
            self.eye.enabled     = True
            self.gesture.enabled = True
            self.gesture.actions_enabled = True
            self.logger.info("System RESUMED by AI decision")

    # ─────────────────────────────────────────────────────────────────────
    #  NOISE FILTER
    # ─────────────────────────────────────────────────────────────────────
    def _is_noise(self, action, buf):
        """Suppress if same action has fired ≥4 times in the last 5 frames."""
        if str(action).startswith("SCROLL("):
            return False
        if not buf:
            return False
        return list(buf).count(action) >= 4

    def _get_buf(self, source):
        return {
            "VOICE":   self._voice_actions,
            "GESTURE": self._gesture_actions,
            "EYE":     self._eye_actions,
        }.get(source, self._eye_actions)

    # ─────────────────────────────────────────────────────────────────────
    #  DRAW STATUS OVERLAY
    # ─────────────────────────────────────────────────────────────────────
    def draw_status(self, frame, action):
        h, w = frame.shape[:2]

        # State banner
        state_color = {
            STATE_ACTIVE:  (0, 220, 120),
            STATE_PAUSED:  (80, 80, 255),
        }.get(self.state, (200, 200, 0))

        cv2.putText(frame, f"[AI] {self.state}  mode={self.active_mode}",
                    (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, state_color, 1)

        if action:
            text = f"ACTION: {action['source']} → {action['action']}"
            cv2.putText(frame, text, (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 220, 0), 1)

        # Priority legend
        legend = "Priority: VOICE(3) > GESTURE(2) > EYE(1)"
        cv2.putText(frame, legend, (w - 340, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

        # Biometric overlay
        if self._bio is not None:
            self._bio.draw_overlay(frame)

    # ─────────────────────────────────────────────────────────────────────
    #  ANALYTICS
    # ─────────────────────────────────────────────────────────────────────
    def get_stats(self):
        source_counts = {"VOICE": 0, "GESTURE": 0, "EYE": 0}
        for evt in self._event_log:
            source_counts[evt["source"]] = source_counts.get(evt["source"], 0) + 1
        return {
            "total_decisions": self.decision_count,
            "source_breakdown": source_counts,
            "last_action":  self.last_action,
            "last_source":  self.last_source,
            "system_state": self.state,
        }
