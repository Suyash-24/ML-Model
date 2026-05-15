"""
modules/gesture_engine.py  —  v4.0  TILT-INVARIANT + FAST-MOTION STABLE
─────────────────────────────────────────────────────────────────────────
Two root causes fixed:

PROBLEM 1 — HAND TILT breaks finger detection
  Old code:   tip.y < pip.y  →  "tip is above pip on screen"
              Breaks at any tilt > ~30° because screen-Y is wrong reference

  Fix:        Rotate ALL landmarks into the hand's own coordinate frame
              first, THEN check finger extension using angles.
              Steps:
                1. Build local axes from wrist → middle_MCP (hand direction)
                   and wrist → index_MCP (palm normal)
                2. Project every landmark onto (local_x, local_y)
                3. In local frame, "finger up" = tip.local_y > pip.local_y
                   regardless of how much the hand is tilted or rotated

PROBLEM 2 — FAST MOVEMENT drops frames + snaps cursor
  Old code:   Rolling mean buffer → freezes on drop, snaps on return
  Fix:        1D Kalman filter per axis
                - During fast motion: filter predicts ahead, no freeze
                - On re-detection: smooth pull back, no snap
                - During brief drops (≤ MAX_DROP_FRAMES): extrapolate
                  from last known velocity instead of resetting

GESTURE MAP (tilt-invariant):
  ✋ OPEN PALM       → Neutral / resume
  ✊ FIST            → Pause system
  ☝  INDEX ONLY     → Move cursor (Kalman-smoothed)
    🤏 PINCH           → Click (short) / Drag (hold)
    ✌  TWO FINGERS    → Smooth scroll (proportional)
    🤟 THREE FINGERS  → Volume control (proportional)
    🖐 FOUR FINGERS   → Volume up (discrete)
  👌 OK              → Double click
  🤙 PINKY+THUMB    → Switch tab

PLACE THIS FILE AT:
  ML-Model/modules/gesture_engine.py   ← replace existing file
"""

import cv2
import numpy as np
import pyautogui
import time
import mediapipe as mp
from utils.logger import EyeconLogger

pyautogui.FAILSAFE = False

# ── MediaPipe landmark IDs ────────────────────────────────────────────────────
WRIST       =  0
THUMB_CMC   =  1; THUMB_MCP  =  2; THUMB_IP   =  3; THUMB_TIP  =  4
INDEX_MCP   =  5; INDEX_PIP  =  6; INDEX_DIP  =  7; INDEX_TIP  =  8
MIDDLE_MCP  =  9; MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP    = 13; RING_PIP   = 14; RING_DIP   = 15; RING_TIP   = 16
PINKY_MCP   = 17; PINKY_PIP  = 18; PINKY_DIP  = 19; PINKY_TIP  = 20

MAX_DROP_FRAMES = 6   # extrapolate for up to this many consecutive missing frames


# ═════════════════════════════════════════════════════════════════════════════
#  KALMAN FILTER  (1-D, constant-velocity model)
#  State: [position, velocity]
# ═════════════════════════════════════════════════════════════════════════════
class Kalman1D:
    """
    Lightweight scalar Kalman filter.
    Works far better than rolling-mean for fast, erratic motion.
    """
    def __init__(self, q=1e-3, r=0.01):
        """
        q — process noise  (higher = trusts measurements more, less smoothing)
        r — measurement noise (higher = more smoothing, more lag)
        """
        self._x  = np.array([0.0, 0.0])   # [pos, vel]
        self._P  = np.eye(2) * 1.0
        self._F  = np.array([[1.0, 1.0],  # state transition (pos += vel)
                              [0.0, 1.0]])
        self._H  = np.array([[1.0, 0.0]]) # observation (we see position)
        self._Q  = np.eye(2) * q           # process noise cov
        self._R  = np.array([[r]])         # measurement noise cov
        self._initialised = False

    def update(self, measurement: float) -> float:
        if not self._initialised:
            self._x[0] = measurement
            self._initialised = True
            return measurement

        # Predict
        x_p = self._F @ self._x
        P_p = self._F @ self._P @ self._F.T + self._Q

        # Update
        S   = self._H @ P_p @ self._H.T + self._R
        K   = P_p @ self._H.T @ np.linalg.inv(S)
        self._x = x_p + (K @ (np.array([[measurement]]) - self._H @ x_p)).flatten()
        self._P = (np.eye(2) - K @ self._H) @ P_p
        return float(self._x[0])

    def predict(self) -> float:
        """Called when measurement is missing — extrapolate from velocity."""
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        return float(self._x[0])

    def reset(self, value: float):
        self._x = np.array([value, 0.0])
        self._P = np.eye(2) * 1.0
        self._initialised = True


# ═════════════════════════════════════════════════════════════════════════════
#  HAND ORIENTATION  —  build local coordinate frame
# ═════════════════════════════════════════════════════════════════════════════
def _local_frame(pts: np.ndarray):
    """
    Build a 2-D local coordinate frame anchored at the wrist.
    Returns pts transformed so that:
      - origin = wrist
      - local_y axis points from wrist toward middle MCP  (palm direction)
      - local_x axis is perpendicular (across palm)
    This makes finger extension checks rotation-invariant.

    Args:
        pts: (21, 2) array of normalised landmark (x, y) positions

    Returns:
        local_pts: (21, 2) array in the local frame
        angle_deg: hand rotation angle in degrees (for debug overlay)
    """
    origin = pts[WRIST].copy()

    # Primary axis: wrist → middle MCP
    primary = pts[MIDDLE_MCP] - origin
    norm = np.linalg.norm(primary)
    if norm < 1e-6:
        return pts - origin, 0.0

    primary /= norm
    # Perpendicular (rotate 90° CCW)
    perp = np.array([-primary[1], primary[0]])

    # Rotation matrix  [perp | primary]  →  columns are local x, local y
    R = np.stack([perp, primary], axis=1)   # shape (2, 2)

    local_pts = (pts - origin) @ R          # project each point
    angle_deg = float(np.degrees(np.arctan2(primary[1], primary[0])))
    return local_pts, angle_deg


# ═════════════════════════════════════════════════════════════════════════════
#  FINGER STATE  (in local frame — tilt-invariant)
# ═════════════════════════════════════════════════════════════════════════════
def _fingers_up_local(lpts: np.ndarray, pts_global: np.ndarray):
    """
    Args:
        lpts:       (21,2) landmarks in local hand frame
        pts_global: (21,2) original normalised landmarks (for thumb side check)

    Returns [thumb, index, middle, ring, pinky]  True = extended
    """
    # Thumb — special case: use angle between thumb vector and index MCP direction
    # because thumb is on the side of the hand, local_y approach doesn't work
    thumb_vec   = pts_global[THUMB_TIP] - pts_global[THUMB_CMC]
    index_vec   = pts_global[INDEX_MCP] - pts_global[WRIST]
    dot         = np.dot(thumb_vec, index_vec)
    thumb_ext   = dot < -0.01   # thumb pointing away from index = extended

    # For other 4 fingers: in local frame, tip.y > pip.y means extended
    # (local_y points "up the palm", so extended finger has larger local_y)
    finger_pairs = [
        (INDEX_TIP,  INDEX_PIP),
        (MIDDLE_TIP, MIDDLE_PIP),
        (RING_TIP,   RING_PIP),
        (PINKY_TIP,  PINKY_PIP),
    ]
    fingers = [thumb_ext]
    for tip, pip in finger_pairs:
        # Use local frame Y — works at any hand rotation
        extended = lpts[tip][1] > lpts[pip][1]
        # Also require the finger to be meaningfully extended (not borderline)
        extension_ratio = (lpts[tip][1] - lpts[pip][1]) / (abs(lpts[MIDDLE_MCP][1]) + 1e-6)
        fingers.append(extended and extension_ratio > -0.05)

    return fingers


# ═════════════════════════════════════════════════════════════════════════════
#  GESTURE ENGINE
# ═════════════════════════════════════════════════════════════════════════════
class GestureEngine:
    """
    Drop-in replacement. Same public interface as previous versions:
      __init__(config, feedback)
      process(frame) → dict
      draw_overlay(frame, data)
      cleanup()
    """

    def __init__(self, config, feedback):
        self.cfg      = config
        self.feedback = feedback
        self.logger   = EyeconLogger("GestureEngine")

        # ── MediaPipe ─────────────────────────────────────────────────
        self._mp_hands = mp.solutions.hands
        self._hands    = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=config.get("gesture_detect_conf", 0.72),
            min_tracking_confidence=config.get("gesture_track_conf",  0.65),
        )
        self._mp_draw  = mp.solutions.drawing_utils
        self._mp_style = mp.solutions.drawing_styles

        self._sw, self._sh = pyautogui.size()

        # ── Kalman filters for cursor (x and y independently) ─────────
        _q = config.get("kalman_process_noise",     8e-4)
        _r = config.get("kalman_measurement_noise", 1.5e-2)
        self._kx = Kalman1D(_q, _r)
        self._ky = Kalman1D(_q, _r)

        # ── Config ────────────────────────────────────────────────────
        self._pinch_thresh       = config.get("pinch_distance_thresh",  0.060)
        self._pinch_hold_frames  = config.get("pinch_hold_frames",        20)
        self._ok_thresh          = config.get("ok_distance_thresh",     0.055)
        self._ok_cd              = config.get("ok_cooldown_frames",        6)
        self._scroll_dz          = config.get("scroll_deadzone",         0.007)
        self._scroll_smooth      = config.get("scroll_smooth",           0.35)
        self._scroll_reset_dz    = config.get("scroll_reset_deadzone",   self._scroll_dz * 0.6)
        self._scroll_step        = config.get("scroll_step",            90)
        self._scroll_trigger     = config.get("scroll_trigger",         0.02)
        self._scroll_trigger_x   = config.get("scroll_trigger_x",       self._scroll_trigger)
        self._scroll_trigger_y   = config.get("scroll_trigger_y",       self._scroll_trigger)
        self._scroll_hold_start  = config.get("scroll_hold_start",      0.03)
        self._scroll_hold_frames = config.get("scroll_hold_interval_frames", 6)
        self._scroll_invert_x    = config.get("scroll_invert_x",         False)
        self._scroll_invert_y    = config.get("scroll_invert_y",         False)
        self._scroll_axis_bias_x = config.get("scroll_axis_bias_x",     1.35)
        self._scroll_use_shift_h = config.get("scroll_use_shift_hscroll", True)
        self._vol_sens           = config.get("volume_sensitivity",        90)
        self._confirm_fist       = config.get("fist_confirm_frames",        8)
        self._confirm_ok         = config.get("ok_confirm_frames",          3)
        self._confirm_scroll     = config.get("gesture_scroll_confirm_frames", 2)
        self._confirm_default    = config.get("gesture_confirm_frames",     4)
        self._global_cd          = config.get("gesture_global_cooldown",   14)
        # Camera usable zone (clip edges to avoid edge jitter)
        self._cam_margin         = config.get("cam_margin",              0.12)

        # ── State ─────────────────────────────────────────────────────
        self._prev_raw      = "NEUTRAL"
        self._confirm_frames = 0
        self._action_cd      = 0
        self._ok_latched     = False

        self.actions_enabled = True

        # Pinch
        self._pinch_start   = None   # (nx, ny) where pinch began
        self._pinch_ref     = None   # rolling reference for proportional scroll
        self._pinch_frames  = 0
        self._drag_active   = False

        # Scroll (two-finger)
        self._scroll_anchor = None
        self._scroll_ref = None
        self._scroll_last = None
        self._scroll_accum = 0.0
        self._scroll_axis = None
        self._scroll_dir = 0
        self._scroll_hold_frames_count = 0

        # Cursor tracking
        self._cursor_active = False

        # Volume reference
        self._vol_ref_y     = None

        # Fast-motion: track consecutive missing frames
        self._drop_frames   = 0
        self._last_pts      = None   # last known global pts

        # Hand tilt angle for overlay
        self._hand_angle    = 0.0

        # Stats
        self.gesture_count  = 0
        self.enabled        = True

        self.logger.info("GestureEngine v4.0 — tilt-invariant + Kalman smoothing")

    # ─────────────────────────────────────────────────────────────────────────
    #  MAIN PROCESS  (called every frame)
    # ─────────────────────────────────────────────────────────────────────────
    def process(self, frame) -> dict:
        if not self.enabled:
            return {"active": False}

        self._action_cd = max(0, self._action_cd - 1)

        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        # ── No hand detected ─────────────────────────────────────────
        if not result.multi_hand_landmarks:
            self._drop_frames += 1

            if (self._drop_frames <= MAX_DROP_FRAMES and self._last_pts is not None
                    and self.actions_enabled and self._cursor_active):
                # Extrapolate cursor using Kalman velocity during brief drops
                sx = int(np.clip(self._kx.predict(), 0, self._sw))
                sy = int(np.clip(self._ky.predict(), 0, self._sh))
                pyautogui.moveTo(sx, sy, duration=0)
                return {"active": False, "hand": False,
                        "gesture": "NEUTRAL", "_dropped": True}

            # Real hand loss — finalise pinch, reset
            self._on_hand_lost()
            return {"active": False, "hand": False, "gesture": "NEUTRAL"}

        # ── Hand found ────────────────────────────────────────────────
        self._drop_frames = 0
        lm_raw = result.multi_hand_landmarks[0].landmark
        lm_mp  = result.multi_hand_landmarks[0]

        # (21, 2) global normalised coords
        pts = np.array([[l.x, l.y] for l in lm_raw], dtype=np.float32)
        self._last_pts = pts

        # Build tilt-invariant local frame
        lpts, self._hand_angle = _local_frame(pts)

        # Finger states in local frame
        fu = _fingers_up_local(lpts, pts)

        # Classify
        raw = self._classify(pts, lpts, fu)

        if raw != "OK":
            self._ok_latched = False

        # Confirm
        if raw == self._prev_raw:
            self._confirm_frames += 1
        else:
            self._confirm_frames = 1
            self._prev_raw = raw

        if raw == "FIST":
            needed = self._confirm_fist
        elif raw == "OK":
            needed = self._confirm_ok
        elif raw == "TWO_FINGERS":
            needed = self._confirm_scroll
        else:
            needed = self._confirm_default
        confirmed = self._confirm_frames >= needed

        action   = None
        executed = False
        if confirmed:
            action, executed = self._execute(raw, pts, lpts, fu)
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
            "lpts":       lpts,
            "hand_angle": self._hand_angle,
            "confidence": min(self._confirm_frames / needed, 1.0),
            "fingers_up": fu,
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  CLASSIFY  (uses local frame — tilt invariant)
    # ─────────────────────────────────────────────────────────────────────────
    def _classify(self, pts, lpts, fu) -> str:
        thumb, idx, mid, ring, pinky = fu
        n_up = sum(fu)

        thumb_vec = pts[THUMB_TIP] - pts[THUMB_CMC]
        index_vec = pts[INDEX_MCP] - pts[WRIST]
        thumb_cos = float(
            np.dot(thumb_vec, index_vec)
            / (np.linalg.norm(thumb_vec) * np.linalg.norm(index_vec) + 1e-6)
        )
        thumb_curl = thumb_cos > 0.2
        palm_len = abs(lpts[MIDDLE_MCP][1]) + 1e-6
        thumb_spread_ratio = abs(lpts[THUMB_TIP][0]) / palm_len

        # ── FIST ─────────────────────────────────────────────────────
        if n_up == 0:
            return "FIST"

        # ── OK  (thumb+index pinched, other fingers up) ──────────────
        d_ok = float(np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP]))
        other_up = int(mid) + int(ring) + int(pinky)
        if other_up >= 2 and d_ok < self._ok_thresh:
            return "OK"

        # ── PINCH (thumb + index close, others curled) ───────────────
        if not mid and not ring and not pinky:
            if d_ok < self._pinch_thresh:
                return "PINCH"

        # ── CURSOR (index only) ───────────────────────────────────────
        if idx and not mid and not ring and not pinky:
            return "CURSOR"

        # ── TWO FINGERS (index + middle) ─────────────────────────────
        if idx and mid and not ring and not pinky:
            return "TWO_FINGERS"

        # ── THREE FINGERS ─────────────────────────────────────────────
        if idx and mid and ring and not pinky and not thumb:
            return "THREE_FINGERS"

        # ── OPEN PALM (prioritize over four fingers) ─────────────────
        if idx and mid and ring and pinky:
            if thumb or thumb_spread_ratio > 0.25:
                return "OPEN_PALM"

        # ── FOUR FINGERS (thumb tucked) ───────────────────────────────
        if idx and mid and ring and pinky and not thumb and thumb_curl and thumb_spread_ratio < 0.18:
            return "FOUR_FINGERS"

        # ── PINKY + THUMB (call me) → switch tab ─────────────────────
        if thumb and pinky and not idx and not mid and not ring:
            return "PINKY_THUMB"

        return "UNKNOWN"

    # ─────────────────────────────────────────────────────────────────────────
    #  EXECUTE
    # ─────────────────────────────────────────────────────────────────────────
    def _execute(self, gesture, pts, lpts, fu):
        cd_ok = self._action_cd == 0

        if not self.actions_enabled:
            if gesture == "OPEN_PALM":
                self._on_hand_lost()
                return "RESUME", True
            return gesture, False

        if gesture != "CURSOR":
            self._cursor_active = False

        if gesture == "OPEN_PALM":
            if not self.actions_enabled:
                self._on_hand_lost()
                return "RESUME", True
            return "OPEN_PALM", False

        if gesture == "FIST":
            if self._drag_active:
                pyautogui.mouseUp(); self._drag_active = False
            if cd_ok:
                self.feedback.speak("System paused")
                self.feedback.beep(280, 120)
                self._action_cd = self._global_cd * 3
                return "PAUSE", True
            return "FIST", False

        if gesture == "CURSOR":
            return self._move_cursor(pts)

        if gesture == "PINCH":
            return self._handle_pinch(pts)

        if gesture == "TWO_FINGERS":
            return self._handle_scroll(pts, lpts)

        if gesture == "THREE_FINGERS":
            return self._handle_volume(pts, lpts)

        if gesture == "FOUR_FINGERS" and cd_ok:
            for _ in range(3):
                pyautogui.press("volumeup")
            self.feedback.beep(660, 70)
            self._action_cd = self._global_cd * 2
            return "VOLUME_UP", True

        if gesture == "OK" and cd_ok:
            if self._ok_latched:
                return "OK", False
            pyautogui.click()
            self.feedback.beep(520, 60)
            self._action_cd = self._ok_cd
            self._ok_latched = True
            return "CLICK", True

        if gesture == "PINKY_THUMB" and cd_ok:
            pyautogui.hotkey("ctrl", "tab")
            self.feedback.beep(600, 80)
            self._action_cd = self._global_cd * 3
            return "SWITCH_TAB", True

        return gesture, False

    # ─────────────────────────────────────────────────────────────────────────
    #  CURSOR MOVE  — Kalman-filtered
    # ─────────────────────────────────────────────────────────────────────────
    def _move_cursor(self, pts):
        self._cursor_active = True
        if self._drag_active:
            pyautogui.mouseUp(); self._drag_active = False

        m = self._cam_margin
        raw_x = float(np.clip((pts[INDEX_TIP][0] - m) / (1.0 - 2*m), 0.0, 1.0))
        raw_y = float(np.clip((pts[INDEX_TIP][1] - m) / (1.0 - 2*m), 0.0, 1.0))

        # Kalman update gives smooth, lag-minimal position
        sx = int(self._kx.update(raw_x * self._sw))
        sy = int(self._ky.update(raw_y * self._sh))
        sx = int(np.clip(sx, 0, self._sw - 1))
        sy = int(np.clip(sy, 0, self._sh - 1))

        pyautogui.moveTo(sx, sy, duration=0)
        return "CURSOR_MOVE", True

    # ─────────────────────────────────────────────────────────────────────────
    #  PINCH  —  click / proportional scroll / drag
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_pinch(self, pts):
        # Pinch centroid in normalised space
        cx = float((pts[THUMB_TIP][0] + pts[INDEX_TIP][0]) / 2)
        cy = float((pts[THUMB_TIP][1] + pts[INDEX_TIP][1]) / 2)

        if self._pinch_start is None:
            self._pinch_start  = (cx, cy)
            self._pinch_ref    = (cx, cy)
            self._pinch_frames = 0
            return "PINCH_START", False

        self._pinch_frames += 1

        # ── Held still long enough → drag ────────────────────────────
        if self._pinch_frames >= self._pinch_hold_frames:
            if not self._drag_active:
                pyautogui.mouseDown()
                self._drag_active = True
                self.feedback.beep(350, 100)
            # Move cursor while dragging
            m  = self._cam_margin
            sx = int(self._kx.update(np.clip((cx - m) / (1-2*m), 0, 1) * self._sw))
            sy = int(self._ky.update(np.clip((cy - m) / (1-2*m), 0, 1) * self._sh))
            pyautogui.moveTo(int(np.clip(sx, 0, self._sw-1)),
                             int(np.clip(sy, 0, self._sh-1)), duration=0)
            return "DRAG", True

        return "PINCH_HOLD", False

    # ─────────────────────────────────────────────────────────────────────────
    #  TWO-FINGER SCROLL  — proportional with accumulation
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_scroll(self, pts, lpts):
        raw_x = float((lpts[INDEX_TIP][0] + lpts[MIDDLE_TIP][0]) / 2.0)
        raw_y = float((lpts[INDEX_TIP][1] + lpts[MIDDLE_TIP][1]) / 2.0)

        if self._scroll_last is None:
            self._scroll_last = (raw_x, raw_y)
        last_x, last_y = self._scroll_last
        filt_x = last_x + (raw_x - last_x) * self._scroll_smooth
        filt_y = last_y + (raw_y - last_y) * self._scroll_smooth
        self._scroll_last = (filt_x, filt_y)

        if self._scroll_anchor is None:
            self._scroll_anchor = (filt_x, filt_y)
            self._scroll_ref = (filt_x, filt_y)
            self._scroll_accum = 0.0
            self._scroll_axis = None
            self._scroll_dir = 0
            self._scroll_hold_frames_count = 0
            return "SCROLL_READY", False

        anchor_x, anchor_y = self._scroll_anchor
        dx = filt_x - anchor_x
        dy = filt_y - anchor_y

        if self._scroll_axis is None:
            if max(abs(dx), abs(dy)) < self._scroll_dz:
                return "TWO_FINGERS", False
            self._scroll_axis = "x" if abs(dx) * self._scroll_axis_bias_x > abs(dy) else "y"
            axis_delta = dx if self._scroll_axis == "x" else dy
            self._scroll_dir = 1 if axis_delta > 0 else -1
            self._scroll_ref = (filt_x, filt_y)
            self._scroll_accum = 0.0
            self._scroll_hold_frames_count = 0
            return "SCROLL_START", False

        axis_anchor_delta = dx if self._scroll_axis == "x" else dy

        if axis_anchor_delta * self._scroll_dir < 0:
            if max(abs(dx), abs(dy)) < self._scroll_reset_dz:
                self._scroll_axis = None
                self._scroll_dir = 0
                self._scroll_anchor = (filt_x, filt_y)
                self._scroll_ref = (filt_x, filt_y)
                self._scroll_accum = 0.0
                self._scroll_hold_frames_count = 0
            else:
                self._scroll_ref = (filt_x, filt_y)
                self._scroll_accum = 0.0
                self._scroll_hold_frames_count = 0
            return "TWO_FINGERS", False

        ref_x, ref_y = self._scroll_ref
        axis_delta = (filt_x - ref_x) if self._scroll_axis == "x" else (filt_y - ref_y)
        self._scroll_accum += axis_delta
        self._scroll_ref = (filt_x, filt_y)

        trigger = self._scroll_trigger_x if self._scroll_axis == "x" else self._scroll_trigger_y
        steps = int(self._scroll_accum / trigger)
        if steps != 0:
            units = steps * self._scroll_step
            if self._scroll_axis == "y":
                if self._scroll_invert_y:
                    units *= -1
                pyautogui.scroll(-units)
            else:
                if self._scroll_invert_x:
                    units *= -1
                self._hscroll(units)
            self._scroll_accum -= steps * trigger
            self._scroll_hold_frames_count = 0
            return f"SCROLL({units})", True

        if abs(axis_anchor_delta) >= self._scroll_hold_start:
            self._scroll_hold_frames_count += 1
            if self._scroll_hold_frames_count >= self._scroll_hold_frames:
                units = self._scroll_step * self._scroll_dir
                if self._scroll_axis == "y":
                    if self._scroll_invert_y:
                        units *= -1
                    pyautogui.scroll(-units)
                else:
                    if self._scroll_invert_x:
                        units *= -1
                    self._hscroll(units)
                self._scroll_hold_frames_count = 0
                return f"SCROLL({units})", True
        else:
            self._scroll_hold_frames_count = 0

        return "TWO_FINGERS", False

    def _hscroll(self, units: int):
        if self._scroll_use_shift_h:
            pyautogui.keyDown("shift")
            pyautogui.scroll(units)
            pyautogui.keyUp("shift")
        else:
            pyautogui.hscroll(units)

    def _release_pinch(self):
        """Called when pinch gesture ends (hand opens or lost)."""
        action = None
        if self._drag_active:
            pyautogui.mouseUp()
            self._drag_active = False
            self.feedback.beep(280, 80)
            action = "DROP"
        self._pinch_start  = None
        self._pinch_ref    = None
        self._pinch_frames = 0
        return action

    # ─────────────────────────────────────────────────────────────────────────
    #  VOLUME  — two fingers, proportional to vertical movement
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_volume(self, pts, lpts):
        # Use average local frame Y of three fingertips — tilt invariant
        ctrl_y = float((lpts[INDEX_TIP][1] + lpts[MIDDLE_TIP][1] + lpts[RING_TIP][1]) / 3.0)

        if self._vol_ref_y is None:
            self._vol_ref_y = ctrl_y
            return "VOL_START", False

        dy = ctrl_y - self._vol_ref_y

        if abs(dy) > 0.010:
            steps = min(int(abs(dy) * self._vol_sens), 6)
            key   = "volumedown" if dy < 0 else "volumeup"
            for _ in range(steps):
                pyautogui.press(key)
            self._vol_ref_y = ctrl_y
            return f"VOL_{'UP' if dy > 0 else 'DOWN'}", True

        return "TWO_FINGERS", False

    # ─────────────────────────────────────────────────────────────────────────
    #  BRIGHTNESS  — three fingers, same pattern as volume
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_brightness(self, pts, lpts):
        ctrl_y = float(lpts[MIDDLE_TIP][1])

        if not hasattr(self, "_bright_ref_y") or self._bright_ref_y is None:
            self._bright_ref_y = ctrl_y
            return "BRIGHT_START", False

        dy = ctrl_y - self._bright_ref_y

        if abs(dy) > 0.015:
            steps = min(int(abs(dy) * 60), 5)
            key   = "brightnessdown" if dy < 0 else "brightnessup"
            for _ in range(steps):
                pyautogui.press(key)
            self._bright_ref_y = ctrl_y
            return f"BRIGHT_{'UP' if dy > 0 else 'DOWN'}", True

        return "THREE_FINGERS", False

    # ─────────────────────────────────────────────────────────────────────────
    #  HAND LOST / RESET
    # ─────────────────────────────────────────────────────────────────────────
    def _on_hand_lost(self):
        if self._pinch_start is not None:
            self._release_pinch()
        self._pinch_start    = None
        self._pinch_ref      = None
        self._pinch_frames   = 0
        self._vol_ref_y      = None
        self._ok_latched     = False
        self._scroll_anchor = None
        self._scroll_ref = None
        self._scroll_last = None
        self._scroll_accum = 0.0
        self._scroll_axis = None
        self._scroll_dir = 0
        self._scroll_hold_frames_count = 0
        self._cursor_active  = False
        if hasattr(self, "_bright_ref_y"):
            self._bright_ref_y = None
        self._drop_frames    = 0

    # ─────────────────────────────────────────────────────────────────────────
    #  OVERLAY
    # ─────────────────────────────────────────────────────────────────────────
    def draw_overlay(self, frame, data):
        h, w = frame.shape[:2]

        if not data.get("active"):
            cv2.putText(frame, "Gesture: NO HAND",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (80, 80, 200), 1)
            return

        # Hand skeleton
        self._mp_draw.draw_landmarks(
            frame, data["landmarks"],
            self._mp_hands.HAND_CONNECTIONS,
            self._mp_style.get_default_hand_landmarks_style(),
            self._mp_style.get_default_hand_connections_style(),
        )

        gesture  = data.get("gesture",   "")
        action   = data.get("action",    "")
        conf     = data.get("confidence", 0.0)
        angle    = data.get("hand_angle", 0.0)
        executed = data.get("executed",  False)

        # ── Confidence bar ────────────────────────────────────────────
        bar_w = int(conf * 140)
        cv2.rectangle(frame, (10, 97), (150, 102), (40, 40, 40), -1)
        bar_col = (0, 220, 100) if conf > 0.8 else (0, 180, 230)
        cv2.rectangle(frame, (10, 97), (10 + bar_w, 102), bar_col, -1)

        # ── Gesture label ─────────────────────────────────────────────
        label = f"Gesture: {gesture}"
        if action and executed:
            label += f"  →  {action}"
        col = (0, 220, 120) if data.get("confirmed") else (180, 180, 50)
        cv2.putText(frame, label, (10, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        # ── Tilt angle ────────────────────────────────────────────────
        cv2.putText(frame, f"tilt={angle:.0f}°",
                    (w - 90, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (120, 120, 120), 1)

        # ── Drag banner ───────────────────────────────────────────────
        if self._drag_active:
            cv2.rectangle(frame, (0, 0), (w, 24), (0, 0, 180), -1)
            cv2.putText(frame, "DRAGGING — open hand to drop",
                        (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)

        # ── Scroll anchor line ────────────────────────────────────────
        if self._pinch_start and self._pinch_ref:
            ax = int(self._pinch_start[0] * w)
            ay = int(self._pinch_start[1] * h)
            rx = int(self._pinch_ref[0]   * w)
            ry = int(self._pinch_ref[1]   * h)
            cv2.line(frame, (ax, ay), (rx, ry), (0, 200, 255), 1)
            cv2.circle(frame, (ax, ay), 5, (0, 200, 255), -1)

        # ── Drop-frame indicator ─────────────────────────────────────
        if data.get("_dropped"):
            cv2.putText(frame, f"PREDICTING ({self._drop_frames}f)",
                        (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 140, 0), 1)

    # ─────────────────────────────────────────────────────────────────────────
    #  CLEANUP
    # ─────────────────────────────────────────────────────────────────────────
    def cleanup(self):
        if self._drag_active:
            pyautogui.mouseUp()
        self._hands.close()
        self.logger.info(
            f"GestureEngine v4.0 closed — "
            f"{self.gesture_count} gestures executed"
        )