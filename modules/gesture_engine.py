"""
modules/gesture_engine.py
─────────────────────────
Hand Gesture Recognition System (MediaPipe Tasks API)

Supported gestures:
  ✊ FIST          → Pause / Stop system
  ✋ OPEN_PALM     → Activate / Resume system
  🤏 PINCH         → Click / Select
  👉 SWIPE_RIGHT   → Switch tab right
  👈 SWIPE_LEFT    → Switch tab left
  ✌  TWO_FINGERS   → Scroll up/down
    3 FINGERS        → Volume down
    4 FINGERS        → Volume up
"""

import cv2
import numpy as np
import pyautogui
import time
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from collections import Counter, deque
from utils.logger import EyeconLogger

# MediaPipe hand landmark IDs
WRIST        = 0
THUMB_TIP    = 4;  THUMB_IP    = 3;  THUMB_MCP = 2
INDEX_TIP    = 8;  INDEX_PIP   = 6
MIDDLE_TIP   = 12; MIDDLE_PIP  = 10
RING_TIP     = 16; RING_PIP    = 14
PINKY_TIP    = 20; PINKY_PIP   = 18

# Hand connection pairs for drawing (replicates mp.solutions.hands.HAND_CONNECTIONS)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),      # thumb
    (0,5),(5,6),(6,7),(7,8),      # index
    (0,9),(9,10),(10,11),(11,12), # middle  (changed: 5→0 for wrist-based)
    (0,13),(13,14),(14,15),(15,16), # ring
    (0,17),(17,18),(18,19),(19,20), # pinky
    (5,9),(9,13),(13,17),         # palm
]


class GestureEngine:
    def __init__(self, config, feedback):
        self.cfg      = config
        self.feedback = feedback
        self.logger   = EyeconLogger("GestureEngine")

        # ── MediaPipe Tasks API: HandLandmarker ─────────────────────────
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "hand_landmarker.task"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand landmarker model not found at {model_path}. "
                "Download from: https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)

        # State
        self.prev_gesture     = None
        self.gesture_frames   = 0
        self.confirm_frames   = self.cfg.get("gesture_confirm_frames", 8)
        self._fist_confirm_frames = int(self.cfg.get("fist_confirm_frames", 6))
        if self._fist_confirm_frames < 1:
            self._fist_confirm_frames = 1
        self.gesture_cooldown = 0
        self.gesture_count    = 0
        smooth_frames = max(1, int(self.cfg.get("gesture_smooth_frames", 4)))
        self._gesture_hist = deque(maxlen=smooth_frames)
        self._gesture_smooth_min = int(self.cfg.get("gesture_smooth_min_count", 3))
        if self._gesture_smooth_min < 1:
            self._gesture_smooth_min = 1
        if self._gesture_smooth_min > self._gesture_hist.maxlen:
            self._gesture_smooth_min = self._gesture_hist.maxlen
        self._gesture_smooth_exclude = {"UNKNOWN", "PINCH"}
        self._hand_present_frames = 0
        self._fist_min_present_frames = int(self.cfg.get("fist_min_present_frames", 6))
        if self._fist_min_present_frames < 0:
            self._fist_min_present_frames = 0
        self._fist_edge_margin = float(self.cfg.get("fist_edge_margin", 0.04))
        if self._fist_edge_margin < 0:
            self._fist_edge_margin = 0.0

        # Swipe tracking
        self._swipe_start_x  = None
        self._swipe_start_y  = None
        self._swipe_start_t  = None
        self.SWIPE_THRESH_PX = 80
        self.SWIPE_THRESH_MS = 600

        # Scroll state
        self._scroll_last_y      = None
        self._scroll_sensitivity = self.cfg.get("gesture_scroll_sensitivity", 1200)
        self._scroll_min_delta   = self.cfg.get("gesture_scroll_min_delta", 0.003)
        self._scroll_smooth      = self.cfg.get("gesture_scroll_smooth", 0.4)
        self._scroll_min_scroll  = self.cfg.get("gesture_scroll_min_scroll", 1)
        self._scroll_confirm_frames = self.cfg.get("gesture_scroll_confirm_frames", 2)
        self._scroll_accum       = 0.0

        # Gesture thresholds
        self._pinch_thresh = self.cfg.get("pinch_distance_thresh", 0.07)
        self._pinch_clear  = self.cfg.get("pinch_clearance_factor", 1.8)
        self._pinch_confirm_frames = self.cfg.get("pinch_confirm_frames", 2)
        self._pinch_cooldown_frames = self.cfg.get("pinch_cooldown_frames", 6)
        self._pinch_frames = 0
        self._pinch_latched = False

        self.enabled = True
        self.actions_enabled = True
        self.logger.info("GestureEngine initialised — 8 gestures mapped (Tasks API)")

    # ─────────────────────────────────────────────────────────────────────
    #  PER-FRAME PROCESSING
    # ─────────────────────────────────────────────────────────────────────
    def process(self, frame):
        if not self.enabled:
            return {"active": False}

        h, w  = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to MediaPipe Image and detect
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.hand_landmarker.detect(mp_image)

        if not result.hand_landmarks:
            self._swipe_start_x = None
            self._scroll_last_y = None
            self._scroll_accum = 0.0
            self._pinch_frames = 0
            self._pinch_latched = False
            self.prev_gesture = None
            self.gesture_frames = 0
            self._gesture_hist.clear()
            self._hand_present_frames = 0
            self.gesture_cooldown = 0
            return {"active": False, "hand": False}

        # Tasks API returns list of NormalizedLandmark lists
        lm_list = result.hand_landmarks[0]
        self._hand_present_frames += 1

        # Normalised landmark array  shape (21, 2)
        pts = np.array([[lm.x, lm.y] for lm in lm_list])

        # Finger states
        fingers_up = self._fingers_up(pts)

        # Raw gesture classify
        raw_gest = self._classify(pts, fingers_up)
        smooth_gest = self._smooth_gesture(raw_gest)
        gest = raw_gest if raw_gest == "PINCH" else smooth_gest
        pinch_dist = np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP])
        pinch_close = pinch_dist < self._pinch_thresh

        if gest != "TWO_FINGERS":
            self._scroll_last_y = None
            self._scroll_accum = 0.0

        # Confirm gesture over N frames to avoid flickering
        action, confirmed = self._confirm_gesture(gest)
        if gest == "PINCH":
            if pinch_close:
                self._pinch_frames += 1
            else:
                self._pinch_frames = 0

            if not self._pinch_latched and self._pinch_frames >= self._pinch_confirm_frames:
                confirmed = True
                self._pinch_latched = True
            else:
                confirmed = False
        else:
            self._pinch_frames = 0
            self._pinch_latched = False

        # Execute if confirmed (continuous for two-finger scroll)
        executed = None
        if confirmed:
            if action == "TWO_FINGERS":
                executed = self._execute(action, pts, w, h)
            elif self.gesture_cooldown == 0:
                executed = self._execute(action, pts, w, h)
                if executed:
                    self.gesture_count += 1
                    cooldown_frames = self._pinch_cooldown_frames if action == "PINCH" \
                        else self.cfg.get("gesture_cooldown_frames", 20)
                    self.gesture_cooldown = cooldown_frames

        self.gesture_cooldown = max(0, self.gesture_cooldown - 1)

        return {
            "active":     True,
            "hand":       True,
            "gesture":    gest,
            "confirmed":  confirmed,
            "action":     action,
            "executed":   executed,
            "fingers_up": fingers_up,
            "landmarks":  lm_list,      # Tasks API landmark list
            "pts":        pts,
            "confidence": result.hand_landmarks[0][0].visibility if hasattr(result.hand_landmarks[0][0], 'visibility') else None,
        }

    # ─────────────────────────────────────────────────────────────────────
    #  FINGER STATE
    # ─────────────────────────────────────────────────────────────────────
    def _fingers_up(self, pts):
        """Return list [thumb, index, middle, ring, pinky] → True = extended."""
        up = []
        # Thumb: compare x (mirrored camera)
        up.append(pts[THUMB_TIP][0] < pts[THUMB_IP][0])
        # Other 4 fingers: tip above PIP
        for tip, pip in [(INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
                         (RING_TIP,  RING_PIP),  (PINKY_TIP,  PINKY_PIP)]:
            up.append(pts[tip][1] < pts[pip][1])
        return up

    # ─────────────────────────────────────────────────────────────────────
    #  SMOOTH GESTURE
    # ─────────────────────────────────────────────────────────────────────
    def _smooth_gesture(self, raw_gest):
        if self._gesture_hist.maxlen <= 1:
            return raw_gest
        self._gesture_hist.append(raw_gest)
        counts = Counter(
            g for g in self._gesture_hist
            if g not in self._gesture_smooth_exclude
        )
        if not counts:
            return raw_gest
        top, count = counts.most_common(1)[0]
        if count >= self._gesture_smooth_min:
            return top
        return raw_gest

    # ─────────────────────────────────────────────────────────────────────
    #  CLASSIFY
    # ─────────────────────────────────────────────────────────────────────
    def _classify(self, pts, fu):
        """Map finger state + geometry → gesture name."""
        thumb, idx, mid, ring, pinky = fu

        pinch_dist = np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP])
        pinch_close = pinch_dist < self._pinch_thresh

        n_up = sum(fu)

        # ── FIST  (all fingers curled) ──────────────────────────────
        if n_up == 0:
            return "FIST"

        # ── OPEN PALM  (all 5 up) ───────────────────────────────────
        if n_up == 5:
            return "OPEN_PALM"

        # ── FOUR FINGERS  (index + middle + ring + pinky up) ────────
        if not thumb and idx and mid and ring and pinky:
            return "FOUR_FINGERS"

        # ── THREE FINGERS  (index + middle + ring up) ───────────────
        if not thumb and idx and mid and ring and not pinky:
            return "THREE_FINGERS"

        # ── PINCH  (thumb + index close) ────────────────────────────
        if pinch_close:
            return "PINCH"

        # ── TWO FINGERS  (index + middle up) ────────────────────────
        if idx and mid and n_up <= 3:
            return "TWO_FINGERS"

        # ── POINTING  (index only) ───────────────────────────────────
        if idx and not mid and not ring and not pinky:
            return "POINTING"

        return "UNKNOWN"

    # ─────────────────────────────────────────────────────────────────────
    #  CONFIRM (debounce)
    # ─────────────────────────────────────────────────────────────────────
    def _confirm_gesture(self, gesture):
        if gesture == self.prev_gesture:
            self.gesture_frames += 1
        else:
            self.prev_gesture  = gesture
            self.gesture_frames = 1

        confirmed = self.gesture_frames == self.confirm_frames
        if gesture == "FIST":
            confirmed = self.gesture_frames == self._fist_confirm_frames
        elif gesture == "TWO_FINGERS":
            confirmed = self.gesture_frames >= self._scroll_confirm_frames
        return gesture, confirmed

    # ─────────────────────────────────────────────────────────────────────
    #  EXECUTE
    # ─────────────────────────────────────────────────────────────────────
    def _execute(self, gesture, pts, frame_w, frame_h):
        """Map confirmed gesture → system action."""
        wrist_x_px = int(pts[WRIST][0] * frame_w)
        wrist_y_px = int(pts[WRIST][1] * frame_h)
        now        = time.time() * 1000   # ms

        if not self.actions_enabled and gesture != "OPEN_PALM":
            return None

        if gesture == "FIST":
            if self._hand_present_frames < self._fist_min_present_frames:
                return None
            margin = self._fist_edge_margin
            if margin > 0:
                min_x = float(np.min(pts[:, 0]))
                max_x = float(np.max(pts[:, 0]))
                min_y = float(np.min(pts[:, 1]))
                max_y = float(np.max(pts[:, 1]))
                if (min_x < margin or max_x > (1.0 - margin) or
                        min_y < margin or max_y > (1.0 - margin)):
                    return None
            self.feedback.speak("System paused")
            self.feedback.visual_flash("pause")
            return "PAUSE"

        if gesture == "OPEN_PALM":
            self.feedback.speak("System active")
            self.feedback.visual_flash("resume")
            return "RESUME"

        if gesture == "PINCH":
            pyautogui.click()
            self.feedback.beep(440, 80)
            return "CLICK"

        if gesture == "TWO_FINGERS":
            tip_y = (pts[INDEX_TIP][1] + pts[MIDDLE_TIP][1]) / 2.0
            if self._scroll_last_y is None:
                self._scroll_last_y = tip_y
                return None

            smoothed = (1.0 - self._scroll_smooth) * self._scroll_last_y + self._scroll_smooth * tip_y
            dy = smoothed - self._scroll_last_y
            self._scroll_last_y = smoothed
            if abs(dy) < self._scroll_min_delta:
                return None

            self._scroll_accum += dy * self._scroll_sensitivity
            step = int(self._scroll_accum)
            if abs(step) < self._scroll_min_scroll:
                return None
            self._scroll_accum -= step
            pyautogui.scroll(step)
            return f"SCROLL({'UP' if step > 0 else 'DOWN'})"

        if gesture == "POINTING":
            # Track swipe via wrist movement
            if self._swipe_start_x is None:
                self._swipe_start_x = wrist_x_px
                self._swipe_start_y = wrist_y_px
                self._swipe_start_t = now
            else:
                dx = wrist_x_px - self._swipe_start_x
                dt = now - self._swipe_start_t
                if dt < self.SWIPE_THRESH_MS and abs(dx) > self.SWIPE_THRESH_PX:
                    self._swipe_start_x = None
                    if dx > 0:
                        pyautogui.hotkey("ctrl", "tab")
                        self.feedback.beep(600, 80)
                        return "SWIPE_RIGHT"
                    else:
                        pyautogui.hotkey("ctrl", "shift", "tab")
                        self.feedback.beep(400, 80)
                        return "SWIPE_LEFT"

        if gesture == "FOUR_FINGERS":
            pyautogui.hotkey("volumeup")
            self.feedback.beep(660, 100)
            return "VOLUME_UP"

        if gesture == "THREE_FINGERS":
            pyautogui.hotkey("volumedown")
            self.feedback.beep(220, 100)
            return "VOLUME_DOWN"

        return None

    # ─────────────────────────────────────────────────────────────────────
    #  OVERLAY
    # ─────────────────────────────────────────────────────────────────────
    def draw_overlay(self, frame, data):
        if not data.get("active"):
            cv2.putText(frame, "Gesture: NO HAND", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 255), 1)
            return

        h, w = frame.shape[:2]
        lm_list = data.get("landmarks")
        if lm_list:
            # Draw hand landmarks manually (Tasks API doesn't have mp_draw)
            pts_px = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]

            # Draw connections
            for i, j in HAND_CONNECTIONS:
                if i < len(pts_px) and j < len(pts_px):
                    cv2.line(frame, pts_px[i], pts_px[j], (0, 220, 0), 2)

            # Draw landmarks
            for px, py in pts_px:
                cv2.circle(frame, (px, py), 4, (0, 220, 120), -1)
                cv2.circle(frame, (px, py), 2, (255, 255, 255), -1)

        g = data.get("gesture", "")
        a = data.get("action",  "")
        label = f"Gesture: {g}"
        if a and data.get("executed"):
            label += f"  →  {a}"
        color = (0, 220, 120) if data.get("confirmed") else (200, 200, 50)
        cv2.putText(frame, label, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def cleanup(self):
        self.hand_landmarker.close()
        self.logger.info(f"GestureEngine closed. Total gestures executed: {self.gesture_count}")
