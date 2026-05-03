"""
modules/eye_tracker.py
──────────────────────
Eye Tracking & Cursor Control Module

Features:
  • Real-time gaze estimation via MediaPipe FaceLandmarker (Tasks API)
  • Gaze-to-screen coordinate mapping
  • Dwell-based clicking (stare ≥ threshold → click)
  • Blink detection (rapid close/open → click)
  • On-screen calibration with 9-point grid
"""

import cv2
import numpy as np
import pyautogui
import time
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from utils.logger import EyeconLogger

# ── Eye landmark indices (MediaPipe 478-point mesh) ─────────────────────────
LEFT_EYE_IDX  = [33,  133, 160, 159, 158, 144, 145, 153]
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 373, 374, 380]
LEFT_IRIS     = [468, 469, 470, 471, 472]
RIGHT_IRIS    = [473, 474, 475, 476, 477]

# EAR threshold: below this = eye closed
EAR_THRESHOLD = 0.21
BLINK_FRAMES  = 2        # consecutive closed frames to register a blink
DWELL_FRAMES  = 25       # ~0.8s at 30fps


class EyeTracker:
    def __init__(self, config, feedback):
        self.cfg      = config
        self.feedback = feedback
        self.logger   = EyeconLogger("EyeTracker")

        # ── MediaPipe Tasks API: FaceLandmarker ─────────────────────────
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "face_landmarker.task"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Face landmarker model not found at {model_path}. "
                "Download from: https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            )

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        self.screen_w, self.screen_h = pyautogui.size()
        pyautogui.FAILSAFE = False

        # Calibration mapping (updated by calibrate())
        self.cal_matrix  = None
        self.cal_offsets = {"x_min": 0.3, "x_max": 0.7,
                            "y_min": 0.3, "y_max": 0.7}

        # State
        self.closed_frames   = 0
        self.dwell_frames    = 0
        self.dwell_pos       = None
        self.last_gaze       = (self.screen_w // 2, self.screen_h // 2)
        self.blink_cooldown  = 0
        self.dwell_cooldown  = 0
        self.blink_count     = 0
        self.click_count     = 0

        # Smoothing buffer
        self._gaze_buf = []
        self._buf_size  = self.cfg.get("gaze_smooth_frames", 6)

        self.enabled = True
        self.logger.info("EyeTracker initialised — iris tracking active (Tasks API)")

    # ─────────────────────────────────────────────────────────────────────
    #  CALIBRATION
    # ─────────────────────────────────────────────────────────────────────
    def calibrate(self, cap):
        """9-point calibration: collect iris positions at known screen points."""
        self.logger.info("Starting calibration…")

        points = [
            (0.15, 0.15), (0.5, 0.15), (0.85, 0.15),
            (0.15, 0.5),  (0.5, 0.5),  (0.85, 0.5),
            (0.15, 0.85), (0.5, 0.85), (0.85, 0.85),
        ]

        cal_w, cal_h = 1280, 720
        cal_win = np.zeros((cal_h, cal_w, 3), dtype=np.uint8)

        raw_iris_pts   = []
        screen_pts     = []

        for i, (px, py) in enumerate(points):
            sx, sy = int(px * cal_w), int(py * cal_h)
            samples = []

            for frame_n in range(60):          # 2 seconds per dot at 30 fps
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)

                # Draw calibration screen
                cal_win[:] = (20, 20, 30)
                cv2.putText(cal_win, f"Calibration  {i+1}/9 — Look at the dot",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (180, 180, 180), 1)
                radius = max(4, int(20 * (1 - frame_n / 60)))
                cv2.circle(cal_win, (sx, sy), 24, (40, 80, 255), 1)
                cv2.circle(cal_win, (sx, sy), radius, (80, 160, 255), -1)
                cv2.imshow("Eyecon — Calibration", cal_win)
                cv2.waitKey(1)

                if frame_n > 20:               # skip first frames for settling
                    iris = self._get_iris_position(frame)
                    if iris is not None:
                        samples.append(iris)

            if samples:
                raw_iris_pts.append(np.mean(samples, axis=0))
                screen_pts.append([px, py])

        cv2.destroyWindow("Eyecon — Calibration")

        if len(raw_iris_pts) >= 4:
            # Fit linear mapping: screen ≈ M × iris + b
            iris_arr   = np.array(raw_iris_pts)
            screen_arr = np.array(screen_pts)
            A = np.c_[iris_arr, np.ones(len(iris_arr))]
            self.cal_matrix, _, _, _ = np.linalg.lstsq(A, screen_arr, rcond=None)
            self.logger.info(f"Calibration complete with {len(raw_iris_pts)} points")
        else:
            self.logger.warning("Not enough calibration data — using defaults")

    # ─────────────────────────────────────────────────────────────────────
    #  PER-FRAME PROCESSING
    # ─────────────────────────────────────────────────────────────────────
    def process(self, frame):
        """Return gaze data dict for this frame."""
        if not self.enabled:
            return {"active": False}

        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to MediaPipe Image and detect
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.face_landmarker.detect(mp_image)

        if not result.face_landmarks:
            return {"active": False, "face": False}

        # Tasks API returns list of NormalizedLandmark objects
        lm = result.face_landmarks[0]

        # ── Iris position (normalised 0–1) ─────────────────────────────
        iris = self._get_iris_from_landmarks(lm)

        # ── EAR (eye aspect ratio) for blink detection ─────────────────
        left_ear  = self._ear(lm, LEFT_EYE_IDX,  w, h)
        right_ear = self._ear(lm, RIGHT_EYE_IDX, w, h)
        avg_ear   = (left_ear + right_ear) / 2.0

        blink_event = self._detect_blink(avg_ear)

        # ── Map iris → screen coords ───────────────────────────────────
        gaze_screen = self._map_to_screen(iris)

        # ── Smooth cursor ──────────────────────────────────────────────
        gaze_smooth = self._smooth(gaze_screen)

        # ── Move cursor ────────────────────────────────────────────────
        if self.cfg.get("eye_moves_cursor", True):
            pyautogui.moveTo(*gaze_smooth, duration=0)

        # ── Dwell click ────────────────────────────────────────────────
        dwell_click = self._detect_dwell(gaze_smooth)

        self.last_gaze = gaze_smooth

        return {
            "active":      True,
            "face":        True,
            "iris_norm":   iris,
            "gaze_screen": gaze_screen,
            "gaze_smooth": gaze_smooth,
            "ear":         avg_ear,
            "blink":       blink_event,
            "dwell_click": dwell_click,
            "landmarks":   lm,
        }

    # ─────────────────────────────────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────────────────────────────────
    def _get_iris_position(self, frame):
        """Quick iris extraction from a raw frame (used during calibration)."""
        h, w = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.face_landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        return self._get_iris_from_landmarks(result.face_landmarks[0])

    def _get_iris_from_landmarks(self, lm):
        """Extract average iris position from landmarks."""
        left_x  = np.mean([lm[i].x for i in LEFT_IRIS])
        left_y  = np.mean([lm[i].y for i in LEFT_IRIS])
        right_x = np.mean([lm[i].x for i in RIGHT_IRIS])
        right_y = np.mean([lm[i].y for i in RIGHT_IRIS])
        return ((left_x + right_x) / 2.0, (left_y + right_y) / 2.0)

    def _ear(self, lm, indices, w, h):
        """Eye Aspect Ratio from 8 landmark indices."""
        pts = np.array([[lm[i].x * w, lm[i].y * h] for i in indices])
        vert1 = np.linalg.norm(pts[2] - pts[6])
        vert2 = np.linalg.norm(pts[3] - pts[5])
        horiz = np.linalg.norm(pts[0] - pts[4])
        return (vert1 + vert2) / (2.0 * horiz + 1e-6)

    def _detect_blink(self, ear):
        self.blink_cooldown = max(0, self.blink_cooldown - 1)
        if ear < EAR_THRESHOLD:
            self.closed_frames += 1
        else:
            if BLINK_FRAMES <= self.closed_frames <= 12 and self.blink_cooldown == 0:
                self.closed_frames = 0
                self.blink_cooldown = 20
                self.blink_count += 1
                if self.cfg.get("blink_click", True):
                    pyautogui.click()
                    self.click_count += 1
                return True
            self.closed_frames = 0
        return False

    def _detect_dwell(self, pos, threshold_px=40):
        self.dwell_cooldown = max(0, self.dwell_cooldown - 1)
        if self.dwell_pos is None:
            self.dwell_pos    = pos
            self.dwell_frames = 0
            return False

        dist = np.hypot(pos[0] - self.dwell_pos[0], pos[1] - self.dwell_pos[1])
        if dist < threshold_px:
            self.dwell_frames += 1
            if self.dwell_frames >= DWELL_FRAMES and self.dwell_cooldown == 0:
                self.dwell_frames  = 0
                self.dwell_cooldown = 45
                self.dwell_pos     = None
                if self.cfg.get("dwell_click", True):
                    pyautogui.click()
                    self.click_count += 1
                return True
        else:
            self.dwell_frames = 0
            self.dwell_pos    = pos
        return False

    def _map_to_screen(self, iris):
        if self.cal_matrix is not None:
            v   = np.array([iris[0], iris[1], 1.0])
            out = v @ self.cal_matrix
            nx, ny = float(out[0]), float(out[1])
        else:
            cal = self.cal_offsets
            nx = (iris[0] - cal["x_min"]) / (cal["x_max"] - cal["x_min"] + 1e-6)
            ny = (iris[1] - cal["y_min"]) / (cal["y_max"] - cal["y_min"] + 1e-6)
        nx = np.clip(nx, 0.0, 1.0)
        ny = np.clip(ny, 0.0, 1.0)
        return (int(nx * self.screen_w), int(ny * self.screen_h))

    def _smooth(self, pos):
        self._gaze_buf.append(pos)
        if len(self._gaze_buf) > self._buf_size:
            self._gaze_buf.pop(0)
        xs = [p[0] for p in self._gaze_buf]
        ys = [p[1] for p in self._gaze_buf]
        return (int(np.mean(xs)), int(np.mean(ys)))

    # ─────────────────────────────────────────────────────────────────────
    #  OVERLAY
    # ─────────────────────────────────────────────────────────────────────
    def draw_overlay(self, frame, data):
        if not data.get("active"):
            cv2.putText(frame, "Eye: NO FACE", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 255), 1)
            return

        h, w = frame.shape[:2]
        if data.get("landmarks"):
            lm = data["landmarks"]
            for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
                x = int(lm[idx].x * w)
                y = int(lm[idx].y * h)
                cv2.circle(frame, (x, y), 1, (0, 200, 255), -1)

        ear_str   = f"EAR: {data['ear']:.2f}"
        blink_str = "BLINK" if data.get("blink") else ""
        dwell_str = "DWELL CLICK" if data.get("dwell_click") else ""
        cv2.putText(frame, f"Eye: TRACKING  {ear_str}  {blink_str}{dwell_str}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 120), 1)

    def cleanup(self):
        self.face_landmarker.close()
        self.logger.info(f"EyeTracker closed. Blinks: {self.blink_count}  Clicks: {self.click_count}")
