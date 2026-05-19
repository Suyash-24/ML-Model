"""
modules/gesture_engine.py  —  v5.0  STABLE
───────────────────────────────────────────
All bugs from previous versions fixed:

  BUG 1 — OPEN_PALM did not resume eye tracking
    Fix: OPEN_PALM returns action="RESUME", executed=True
         ai_decision.execute() catches this and re-enables eye tracker

  BUG 2 — Cursor drifted to bottom-right when hand absent
    Fix: Kalman predict() NEVER calls pyautogui.moveTo()
         Only CURSOR gesture with hand present moves the mouse

  BUG 3 — Volume fired randomly
    Fix: Volume is THUMBS_UP/DOWN only (very distinct, rare false positive)
         TWO_FINGERS = scroll only — no persistent reference to drift

  BUG 4 — TWO_FINGERS scroll detected but didn't scroll
    Fix: Frame-delta scroll — delta between consecutive frames, no reference drift
         Tiny deadzone (0.003) so very small movements register

  BUG 5 — Pinch inconsistent
    Fix: Adaptive threshold = pinch_scale_factor × hand_scale (wrist→middle_MCP)
         Scales to actual hand size in frame, not fixed screen ratio

GESTURE MAP:
  ✋ OPEN_PALM   → RESUME system
  ✊ FIST        → PAUSE system
  ☝  INDEX_ONLY → Move cursor (Kalman-filtered)
  🤏 PINCH still → Left click
  🤏 PINCH move  → Scroll proportional (vertical priority)
  🤏 PINCH hold  → Drag & drop
  ✌  TWO_FINGERS → Scroll (frame-delta, trackpad-style)
  👍 THUMBS_UP   → Volume +3
  👎 THUMBS_DOWN → Volume -3
  👌 OK          → Double click

PLACE AT: ML-Model/modules/gesture_engine.py  (replace existing)
"""

import cv2
import numpy as np
import pyautogui
import time
import mediapipe as mp
from utils.logger import EyeconLogger

pyautogui.FAILSAFE = False

# ── Landmark IDs ──────────────────────────────────────────────────────────────
WRIST      =  0
THUMB_CMC  =  1; THUMB_MCP =  2; THUMB_IP  =  3; THUMB_TIP  =  4
INDEX_MCP  =  5; INDEX_PIP =  6; INDEX_DIP =  7; INDEX_TIP  =  8
MIDDLE_MCP =  9; MIDDLE_PIP= 10; MIDDLE_DIP= 11; MIDDLE_TIP = 12
RING_MCP   = 13; RING_PIP  = 14; RING_DIP  = 15; RING_TIP   = 16
PINKY_MCP  = 17; PINKY_PIP = 18; PINKY_DIP = 19; PINKY_TIP  = 20

_MAX_DROP = 4   # extrapolate for up to this many missing frames


# ═════════════════════════════════════════════════════════════════════════════
#  KALMAN 1-D  (position + velocity)
# ═════════════════════════════════════════════════════════════════════════════
class _K1D:
    def __init__(self, q=5e-4, r=8e-3):
        self._x = np.zeros(2)
        self._P = np.eye(2)
        self._F = np.array([[1., 1.], [0., 1.]])
        self._H = np.array([[1., 0.]])
        self._Q = np.eye(2) * q
        self._R = np.array([[r]])
        self._ok = False

    def update(self, z: float) -> float:
        if not self._ok:
            self._x[0] = z; self._ok = True; return z
        xp = self._F @ self._x
        Pp = self._F @ self._P @ self._F.T + self._Q
        K  = (Pp @ self._H.T) / float(self._H @ Pp @ self._H.T + self._R)
        self._x = xp + K.flatten() * (z - float(self._H @ xp))
        self._P = (np.eye(2) - np.outer(K, self._H)) @ Pp
        return float(self._x[0])

    def predict(self) -> float:
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        return float(self._x[0])


# ═════════════════════════════════════════════════════════════════════════════
#  FINGER STATE  — angle-based, tilt-invariant
# ═════════════════════════════════════════════════════════════════════════════
def _fingers_up(pts: np.ndarray):
    """
    Returns [thumb, index, middle, ring, pinky]  True = extended.
    Uses PIP angle for fingers (works at any hand rotation).
    Uses cross-product for thumb (avoids mirroring issues).
    """
    def _angle(a, b, c):
        v1 = a - b; v2 = c - b
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

    # Thumb
    palm   = pts[MIDDLE_MCP] - pts[WRIST]
    thumb  = pts[THUMB_TIP]  - pts[THUMB_MCP]
    cross  = palm[0]*thumb[1] - palm[1]*thumb[0]
    thumb_ext = cross > 0.002

    fingers = [thumb_ext]
    for mcp, pip, tip in [
        (INDEX_MCP,  INDEX_PIP,  INDEX_TIP),
        (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
        (RING_MCP,   RING_PIP,   RING_TIP),
        (PINKY_MCP,  PINKY_PIP,  PINKY_TIP),
    ]:
        fingers.append(_angle(pts[mcp], pts[pip], pts[tip]) > 155.0)

    return fingers


# ═════════════════════════════════════════════════════════════════════════════
#  GESTURE ENGINE
# ═════════════════════════════════════════════════════════════════════════════
class GestureEngine:

    def __init__(self, config, feedback):
        self.cfg      = config
        self.feedback = feedback
        self.logger   = EyeconLogger("GestureEngine")

        self._mp_hands = mp.solutions.hands
        self._hands    = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=config.get("gesture_detect_conf", 0.70),
            min_tracking_confidence=config.get("gesture_track_conf",   0.60),
        )
        self._draw  = mp.solutions.drawing_utils
        self._style = mp.solutions.drawing_styles

        self._sw, self._sh = pyautogui.size()

        # Kalman for cursor
        q = config.get("kalman_q", 5e-4)
        r = config.get("kalman_r", 8e-3)
        self._kx = _K1D(q, r)
        self._ky = _K1D(q, r)

        # Config
        self._pinch_k       = config.get("pinch_scale_factor",    0.28)
        self._pinch_hold_f  = config.get("pinch_hold_frames",       22)
        self._scroll_sens   = config.get("scroll_sensitivity",      700)
        self._scroll_dz     = config.get("scroll_deadzone",       0.010)
        self._margin        = config.get("cam_margin",             0.10)
        self._confirm_fist  = config.get("fist_confirm_frames",       9)
        self._confirm_def   = config.get("gesture_confirm_frames",    4)
        self._confirm_palm  = config.get("palm_confirm_frames",       3)
        self._gcd           = config.get("gesture_global_cooldown",  12)

        # State
        self._prev_raw    = "NONE"
        self._cf          = 0
        self._action_cd   = 0

        # Pinch
        self._pinch_start = None
        self._pinch_ref   = None
        self._pinch_f     = 0
        self._drag        = False

        # Scroll (two-finger frame-delta)
        self._scroll2_prev_y = None

        # Drop-frame tracking
        self._drop_f  = 0
        self._last_pts = None

        self.enabled       = True
        self.paused        = False
        self.gesture_count = 0

        self.logger.info("GestureEngine v5.0 ready")

    # ─────────────────────────────────────────────────────────────────────────
    #  PROCESS  — called every frame
    # ─────────────────────────────────────────────────────────────────────────
    def process(self, frame) -> dict:
        if not self.enabled:
            return {"active": False, "gesture": "DISABLED"}

        self._action_cd = max(0, self._action_cd - 1)
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res  = self._hands.process(rgb)

        # ── No hand ───────────────────────────────────────────────────
        if not res.multi_hand_landmarks:
            self._drop_f += 1
            if self._drop_f <= _MAX_DROP and self._last_pts is not None:
                # Extrapolate in Kalman — DO NOT move mouse
                self._kx.predict()
                self._ky.predict()
            else:
                self._on_hand_lost()
            return {"active": False, "hand": False, "gesture": "NONE"}

        # ── Hand present ──────────────────────────────────────────────
        self._drop_f   = 0
        lm_raw = res.multi_hand_landmarks[0].landmark
        lm_mp  = res.multi_hand_landmarks[0]
        pts    = np.array([[l.x, l.y] for l in lm_raw], dtype=np.float32)
        self._last_pts = pts

        # Adaptive pinch threshold
        hand_scale   = float(np.linalg.norm(pts[MIDDLE_MCP] - pts[WRIST]))
        pinch_thresh = self._pinch_k * hand_scale

        fu  = _fingers_up(pts)
        raw = self._classify(pts, fu, pinch_thresh)

        # Confirm
        if raw == self._prev_raw:
            self._cf += 1
        else:
            self._cf       = 1
            self._prev_raw = raw
            if raw != "TWO_FINGERS":
                self._scroll2_prev_y = None

        needed    = (self._confirm_fist if raw == "FIST"
                     else self._confirm_palm if raw == "OPEN_PALM"
                     else self._confirm_def)
        confirmed = self._cf >= needed

        action, executed = None, False
        if confirmed:
            action, executed = self._execute(raw, pts, fu, pinch_thresh)
            if executed:
                self.gesture_count += 1

        return {
            "active":     True,
            "hand":       True,
            "gesture":    raw,
            "confirmed":  confirmed,
            "action":     action,
            "executed":   executed,
            "landmarks":  lm_mp,
            "pts":        pts,
            "confidence": min(self._cf / max(needed, 1), 1.0),
            "fingers_up": fu,
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  CLASSIFY
    # ─────────────────────────────────────────────────────────────────────────
    def _classify(self, pts, fu, pinch_thresh) -> str:
        thumb, idx, mid, ring, pinky = fu
        n_up = sum(fu)

        if n_up == 0:
            return "FIST"
        if n_up >= 4:
            return "OPEN_PALM"

        # Thumbs up/down (thumb only)
        if thumb and not idx and not mid and not ring and not pinky:
            if pts[THUMB_TIP][1] < pts[THUMB_MCP][1] - 0.04:
                return "THUMBS_UP"
            if pts[THUMB_TIP][1] > pts[THUMB_MCP][1] + 0.04:
                return "THUMBS_DOWN"

        # Pinch (thumb+index close, others curled)
        if not mid and not ring and not pinky:
            d = float(np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP]))
            if d < pinch_thresh:
                return "PINCH"

        # OK (mid+ring+pinky up, thumb+index pinched)
        if mid and ring and pinky and not idx:
            d = float(np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP]))
            if d < pinch_thresh * 1.35:
                return "OK"

        # Index only (cursor)
        if idx and not mid and not ring and not pinky:
            return "INDEX_ONLY"

        # Two fingers (index + middle)
        if idx and mid and not ring and not pinky:
            return "TWO_FINGERS"

        # Three fingers
        if idx and mid and ring and not pinky:
            return "THREE_FINGERS"

        return "UNKNOWN"

    # ─────────────────────────────────────────────────────────────────────────
    #  EXECUTE
    # ─────────────────────────────────────────────────────────────────────────
    def _execute(self, gesture, pts, fu, pinch_thresh):
        cd_ok = self._action_cd == 0

        # OPEN_PALM → RESUME  (KEY FIX: executed=True so ai_decision catches it)
        if gesture == "OPEN_PALM":
            self._reset_dynamic()
            if self.paused and cd_ok:
                self.paused = False
                self.feedback.beep(500, 80)
                self._action_cd = self._gcd * 2
                return "RESUME", True
            return "OPEN_PALM", False

        # FIST → PAUSE
        if gesture == "FIST":
            if self._drag:
                pyautogui.mouseUp(); self._drag = False
            if not self.paused and cd_ok:
                self.paused = True
                self.feedback.beep(280, 120)
                self._action_cd = self._gcd * 3
                return "PAUSE", True
            return "FIST", False

        # All below blocked when paused
        if self.paused:
            return gesture, False

        if gesture == "INDEX_ONLY":
            return self._move_cursor(pts)

        if gesture == "PINCH":
            return self._handle_pinch(pts)

        if gesture == "TWO_FINGERS":
            return self._handle_scroll_two(pts)

        if gesture == "THUMBS_UP" and cd_ok:
            for _ in range(3): pyautogui.press("volumeup")
            self.feedback.beep(660, 70)
            self._action_cd = self._gcd * 2
            return "VOLUME_UP", True

        if gesture == "THUMBS_DOWN" and cd_ok:
            for _ in range(3): pyautogui.press("volumedown")
            self.feedback.beep(220, 70)
            self._action_cd = self._gcd * 2
            return "VOLUME_DOWN", True

        if gesture == "OK" and cd_ok:
            pyautogui.doubleClick()
            self.feedback.beep(520, 60)
            self._action_cd = self._gcd * 3
            return "DOUBLE_CLICK", True

        return gesture, False

    # ─────────────────────────────────────────────────────────────────────────
    #  CURSOR
    # ─────────────────────────────────────────────────────────────────────────
    def _move_cursor(self, pts):
        if self._drag:
            pyautogui.mouseUp(); self._drag = False

        m     = self._margin
        raw_x = float(np.clip((pts[INDEX_TIP][0] - m) / (1.0 - 2*m), 0, 1))
        raw_y = float(np.clip((pts[INDEX_TIP][1] - m) / (1.0 - 2*m), 0, 1))

        sx = int(np.clip(self._kx.update(raw_x * self._sw), 0, self._sw - 1))
        sy = int(np.clip(self._ky.update(raw_y * self._sh), 0, self._sh - 1))
        pyautogui.moveTo(sx, sy, duration=0)
        return "CURSOR_MOVE", True

    # ─────────────────────────────────────────────────────────────────────────
    #  PINCH  — click / proportional scroll / drag
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_pinch(self, pts):
        cx = float((pts[THUMB_TIP][0] + pts[INDEX_TIP][0]) / 2)
        cy = float((pts[THUMB_TIP][1] + pts[INDEX_TIP][1]) / 2)

        if self._pinch_start is None:
            self._pinch_start = (cx, cy)
            self._pinch_ref   = (cx, cy)
            self._pinch_f     = 0
            return "PINCH_START", False

        self._pinch_f += 1

        total_dy = cy - self._pinch_start[1]
        total_dx = cx - self._pinch_start[0]
        dy       = cy - self._pinch_ref[1]
        dx       = cx - self._pinch_ref[0]

        # Vertical scroll (priority)
        if abs(total_dy) > self._scroll_dz:
            if abs(dy) > self._scroll_dz * 0.4:
                amt = int(-dy * self._scroll_sens)
                if amt: pyautogui.scroll(amt)
                self._pinch_ref = (cx, cy)
            return "SCROLL_V", True

        # Horizontal scroll
        if abs(total_dx) > self._scroll_dz * 1.8 and abs(total_dy) < self._scroll_dz * 0.5:
            if abs(dx) > self._scroll_dz * 0.4:
                pyautogui.hscroll(int(dx * 300))
                self._pinch_ref = (cx, cy)
            return "SCROLL_H", True

        # Drag
        if self._pinch_f >= self._pinch_hold_f:
            if not self._drag:
                pyautogui.mouseDown()
                self._drag = True
                self.feedback.beep(350, 100)
            m  = self._margin
            sx = int(np.clip(self._kx.update(
                np.clip((cx - m) / (1 - 2*m), 0, 1) * self._sw), 0, self._sw - 1))
            sy = int(np.clip(self._ky.update(
                np.clip((cy - m) / (1 - 2*m), 0, 1) * self._sh), 0, self._sh - 1))
            pyautogui.moveTo(sx, sy, duration=0)
            return "DRAG", True

        return "PINCH_HOLD", False

    def _release_pinch(self):
        if self._drag:
            pyautogui.mouseUp()
            self._drag = False
            self.feedback.beep(280, 80)
        elif self._pinch_f is not None and self._pinch_f < self._pinch_hold_f:
            if self._action_cd == 0 and not self.paused:
                pyautogui.click()
                self.feedback.beep(480, 55)
                self._action_cd = self._gcd
                self.gesture_count += 1
        self._pinch_start = self._pinch_ref = None
        self._pinch_f     = 0

    # ─────────────────────────────────────────────────────────────────────────
    #  TWO-FINGER SCROLL  — frame-delta (trackpad-style)
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_scroll_two(self, pts):
        """
        Use the Y delta between this frame and the previous.
        No drifting reference point. Works like a physical trackpad.
        """
        cur_y = float((pts[INDEX_TIP][1] + pts[MIDDLE_TIP][1]) / 2)

        if self._scroll2_prev_y is None:
            self._scroll2_prev_y = cur_y
            return "SCROLL_READY", False

        dy = cur_y - self._scroll2_prev_y
        self._scroll2_prev_y = cur_y

        if abs(dy) > 0.003:
            amt = int(-dy * self._scroll_sens)
            if amt: pyautogui.scroll(amt)
            return f"SCROLL_{'DOWN' if dy > 0 else 'UP'}", True

        return "TWO_FINGERS", False

    # ─────────────────────────────────────────────────────────────────────────
    #  RESET
    # ─────────────────────────────────────────────────────────────────────────
    def _on_hand_lost(self):
        if self._pinch_start is not None:
            self._release_pinch()
        self._reset_dynamic()

    def _reset_dynamic(self):
        self._pinch_start    = None
        self._pinch_ref      = None
        self._pinch_f        = 0
        self._scroll2_prev_y = None
        self._drop_f         = 0
        if self._drag:
            pyautogui.mouseUp()
            self._drag = False

    # ─────────────────────────────────────────────────────────────────────────
    #  OVERLAY
    # ─────────────────────────────────────────────────────────────────────────
    def draw_overlay(self, frame, data):
        h, w = frame.shape[:2]

        if not data.get("active"):
            msg = "PAUSED — open palm to resume" if self.paused else "No hand"
            col = (60, 60, 180) if self.paused else (60, 60, 60)
            cv2.putText(frame, f"Gesture: {msg}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
            return

        self._draw.draw_landmarks(
            frame, data["landmarks"],
            self._mp_hands.HAND_CONNECTIONS,
            self._style.get_default_hand_landmarks_style(),
            self._style.get_default_hand_connections_style(),
        )

        gesture  = data.get("gesture",   "")
        action   = data.get("action",    "")
        conf     = data.get("confidence", 0.0)
        executed = data.get("executed",  False)
        fu       = data.get("fingers_up", [])

        # Confidence bar
        bw = int(conf * 140)
        cv2.rectangle(frame, (10, 98), (150, 103), (30, 30, 30), -1)
        cv2.rectangle(frame, (10, 98), (10 + bw, 103),
                      (0, 200, 80) if conf > 0.8 else (0, 140, 220), -1)

        label = f"Gesture: {gesture}"
        if action and executed:
            label += f"  →  {action}"
        col = (0, 210, 110) if data.get("confirmed") else (160, 160, 50)
        cv2.putText(frame, label, (10, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

        # Finger debug  T I M R P
        if fu:
            for i, (n, up) in enumerate(zip("TIMRP", fu)):
                c = (0, 200, 80) if up else (50, 50, 50)
                cv2.putText(frame, n, (10 + i*16, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)

        if self._drag:
            cv2.rectangle(frame, (0, 0), (w, 22), (0, 0, 140), -1)
            cv2.putText(frame, "DRAGGING — open hand to drop",
                        (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                        (255, 255, 255), 1)

        if self.paused:
            cv2.rectangle(frame, (0, 0), (w, 22), (0, 0, 100), -1)
            cv2.putText(frame, "PAUSED — show open palm",
                        (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                        (80, 140, 255), 1)

    # ─────────────────────────────────────────────────────────────────────────
    def cleanup(self):
        if self._drag:
            pyautogui.mouseUp()
        self._hands.close()
        self.logger.info(f"GestureEngine v5.0 — {self.gesture_count} gestures")