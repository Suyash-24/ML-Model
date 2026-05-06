"""
modules/biometric_enroller.py
──────────────────────────────
Biometric Enrollment Module — runs ONCE at registration

What it captures:
  FACE  — 30 frames via MediaPipe FaceMesh
          468 landmarks × (x,y,z) = 1404 floats
          Averaged across frames → stored as mean embedding
          Also computes 20 face-ratio features for faster runtime check

  HAND  — User shows 5 poses (open palm, fist, pinch, point, peace)
          10 frames per pose → landmark distance ratios
          Yields a 15-float hand-signature (pose-invariant, person-specific)

  VOICE — User speaks 3 phrases shown on screen
          Recorded via sounddevice, MFCC extracted via scipy
          Mean of 13 MFCC coefficients stored as voiceprint vector

All embeddings stored as comma-separated floats in data/biometrics.csv
No raw audio/video is ever saved — only the mathematical vectors.
"""

import cv2
import numpy as np
import time
import threading
import csv
import os
import json

import mediapipe as mp

try:
    import sounddevice as sd
    from scipy.fft import dct
    _AUDIO_OK = True
except ImportError:
    _AUDIO_OK = False

from utils.logger import EyeconLogger

logger = EyeconLogger("Enroller")

# ── MediaPipe ─────────────────────────────────────────────────────────────────
_mp_face  = mp.solutions.face_mesh
_mp_hands = mp.solutions.hands

# ── Key landmark indices ──────────────────────────────────────────────────────
# Face ratios
_L_EYE_OUTER  = 33;  _L_EYE_INNER  = 133
_R_EYE_OUTER  = 362; _R_EYE_INNER  = 263
_NOSE_TIP     = 4;   _CHIN         = 152
_FACE_L       = 234; _FACE_R       = 454
_L_BROW       = 70;  _R_BROW       = 300
_MOUTH_L      = 61;  _MOUTH_R      = 291

# Hand landmark IDs
_WRIST   = 0
_TIPS    = [4, 8, 12, 16, 20]   # thumb→pinky tips
_MIDS    = [3, 7, 11, 15, 19]   # mid phalanges
_BASES   = [1, 5, 9, 13, 17]    # MCP joints

# ─────────────────────────────────────────────────────────────────────────────
#  FACE EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────
def _extract_face_embedding(lm, w, h):
    """
    Returns a 1404-float normalised landmark embedding
    AND a 20-float ratio vector.
    """
    # Full normalised landmark positions (1404 floats)
    pts = np.array([[lm[i].x, lm[i].y, lm[i].z] for i in range(468)],
                   dtype=np.float32)
    # Normalise by face bounding box so scale/position invariant
    mn = pts.min(axis=0);  mx = pts.max(axis=0)
    span = mx - mn + 1e-6
    normed = ((pts - mn) / span).flatten()   # 1404 floats, all in [0,1]

    # Also compute 20 geometric ratios for fast runtime cosine check
    def _dist(i, j):
        return np.linalg.norm(
            np.array([lm[i].x, lm[i].y]) - np.array([lm[j].x, lm[j].y])
        )

    face_w   = _dist(_FACE_L, _FACE_R)      + 1e-6
    face_h   = _dist(_NOSE_TIP, _CHIN)      + 1e-6
    eye_dist = _dist(_L_EYE_OUTER, _R_EYE_OUTER)

    ratios = np.array([
        eye_dist / face_w,
        _dist(_L_EYE_OUTER, _L_EYE_INNER) / face_w,
        _dist(_R_EYE_OUTER, _R_EYE_INNER) / face_w,
        _dist(_L_BROW, _R_BROW) / face_w,
        _dist(_NOSE_TIP, _CHIN) / face_w,
        _dist(_MOUTH_L, _MOUTH_R) / face_w,
        lm[_NOSE_TIP].y / (lm[_CHIN].y + 1e-6),
        lm[_L_BROW].y  / (lm[_CHIN].y + 1e-6),
        _dist(_FACE_L, _FACE_R) / (_dist(_NOSE_TIP, _CHIN) + 1e-6),
        lm[_L_EYE_OUTER].x,
        lm[_R_EYE_OUTER].x,
        lm[_NOSE_TIP].x,
        lm[_MOUTH_L].x,
        lm[_MOUTH_R].x,
        lm[_L_EYE_OUTER].y,
        lm[_R_EYE_OUTER].y,
        lm[_NOSE_TIP].y,
        eye_dist / face_h,
        _dist(_MOUTH_L, _MOUTH_R) / eye_dist,
        lm[_CHIN].y - lm[_NOSE_TIP].y,
    ], dtype=np.float32)

    return normed, ratios


def capture_face_embedding(cap, face_mesh, n_frames=30, progress_cb=None):
    """
    Capture n_frames of face data from the webcam.
    Returns (mean_embedding_1404, mean_ratios_20) or (None, None) on failure.
    """
    embeddings = []
    ratios_list = []
    frame_idx = 0

    logger.info("Capturing face embedding…")

    while len(embeddings) < n_frames:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame  = cv2.flip(frame, 1)
        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            emb, rat = _extract_face_embedding(lm, w, h)
            embeddings.append(emb)
            ratios_list.append(rat)

        frame_idx += 1
        if progress_cb:
            progress_cb("face", len(embeddings), n_frames, frame)

        if frame_idx > n_frames * 4:   # timeout guard
            break

    if len(embeddings) < 10:
        logger.warning("Not enough face frames captured")
        return None, None

    mean_emb    = np.mean(embeddings,    axis=0)
    mean_ratios = np.mean(ratios_list,   axis=0)
    logger.info(f"Face embedding captured ({len(embeddings)} frames)")
    return mean_emb, mean_ratios


# ─────────────────────────────────────────────────────────────────────────────
#  HAND SIGNATURE
# ─────────────────────────────────────────────────────────────────────────────
_HAND_POSES = [
    ("Open palm  — hold still", 10),
    ("Fist  — close your hand", 10),
    ("Pinch  — touch thumb + index", 10),
    ("Point  — index finger out", 10),
    ("Peace  — V shape (index + middle)", 10),
]


def _hand_signature(lm):
    """
    Compute a 15-float pose-invariant hand signature from landmark ratios.
    All distances normalised by palm diagonal (wrist → middle MCP).
    """
    pts = np.array([[l.x, l.y] for l in lm], dtype=np.float32)

    palm_diag = np.linalg.norm(pts[_WRIST] - pts[9]) + 1e-6

    sigs = []
    for tip, base in zip(_TIPS, _BASES):
        # Finger extension ratio: tip-to-wrist / palm
        sigs.append(np.linalg.norm(pts[tip] - pts[_WRIST]) / palm_diag)
        # Finger curl: tip-to-base / palm
        sigs.append(np.linalg.norm(pts[tip] - pts[base]) / palm_diag)
        # Mid phalange
        mid_idx = _MIDS[_TIPS.index(tip)]
        sigs.append(np.linalg.norm(pts[mid_idx] - pts[base]) / palm_diag)

    return np.array(sigs[:15], dtype=np.float32)


def capture_hand_signature(cap, hands_detector, progress_cb=None):
    """
    Ask user to hold 5 hand poses × 10 frames each.
    Returns mean 15-float signature or None.
    """
    all_sigs = []
    logger.info("Capturing hand signature…")

    for pose_name, n_frames in _HAND_POSES:
        pose_sigs = []
        frame_idx = 0
        logger.info(f"  Pose: {pose_name}")

        while len(pose_sigs) < n_frames:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame  = cv2.flip(frame, 1)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands_detector.process(rgb)

            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0].landmark
                sig = _hand_signature(lm)
                pose_sigs.append(sig)

            frame_idx += 1
            if progress_cb:
                progress_cb("hand", len(pose_sigs), n_frames, frame, pose_name)

            if frame_idx > n_frames * 5:
                break

        if pose_sigs:
            all_sigs.extend(pose_sigs)

    if not all_sigs:
        logger.warning("No hand data captured")
        return None

    # Mean across ALL poses — gives person's unique hand geometry
    mean_sig = np.mean(all_sigs, axis=0)
    logger.info(f"Hand signature captured ({len(all_sigs)} frames)")
    return mean_sig


# ─────────────────────────────────────────────────────────────────────────────
#  VOICE MFCC
# ─────────────────────────────────────────────────────────────────────────────
_VOICE_PHRASES = [
    "Eyecon start",
    "Open Chrome",
    "Scroll down",
]

_SR   = 16000   # sample rate
_DUR  = 2.5     # seconds per phrase


def _compute_mfcc(audio, sr=_SR, n_mfcc=13, n_fft=512, hop=160):
    """
    Minimal MFCC computation using only numpy + scipy.fft (no librosa needed).
    Returns mean of n_mfcc coefficients.
    """
    audio = audio.astype(np.float32)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)   # normalise

    # Pre-emphasis
    preemph = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # Frame into overlapping windows
    n_frames = 1 + (len(preemph) - n_fft) // hop
    frames   = np.stack([
        preemph[i*hop : i*hop + n_fft] for i in range(n_frames)
    ])
    frames  *= np.hamming(n_fft)

    # Power spectrum
    power = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2

    # Mel filterbank (26 filters, 0–8000 Hz)
    n_mel  = 26
    f_min  = 0.0
    f_max  = sr / 2.0

    def _hz_to_mel(f): return 2595 * np.log10(1 + f / 700)
    def _mel_to_hz(m): return 700 * (10 ** (m / 2595) - 1)

    mel_pts = np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mel + 2)
    hz_pts  = _mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    filterbank = np.zeros((n_mel, n_fft // 2 + 1))
    for m in range(1, n_mel + 1):
        f_m_minus = bin_pts[m - 1]
        f_m       = bin_pts[m]
        f_m_plus  = bin_pts[m + 1]
        for k in range(f_m_minus, f_m):
            filterbank[m-1, k] = (k - f_m_minus) / (f_m - f_m_minus + 1e-8)
        for k in range(f_m, f_m_plus):
            filterbank[m-1, k] = (f_m_plus - k) / (f_m_plus - f_m + 1e-8)

    mel_energy = np.dot(power, filterbank.T)
    log_mel    = np.log(mel_energy + 1e-8)

    # DCT to get MFCCs
    mfcc = dct(log_mel, type=2, axis=1, norm='ortho')[:, :n_mfcc]

    return np.mean(mfcc, axis=0).astype(np.float32)   # 13 floats


def capture_voice_mfcc(progress_cb=None):
    """
    Record 3 phrases and return mean 13-float MFCC vector.
    Returns None if audio not available.
    """
    if not _AUDIO_OK:
        logger.warning("sounddevice not available — skipping voice enrollment")
        return np.zeros(13, dtype=np.float32)   # zero vector = voice disabled

    all_mfccs = []
    logger.info("Capturing voice MFCC…")

    for i, phrase in enumerate(_VOICE_PHRASES):
        logger.info(f"  Phrase {i+1}: '{phrase}'")
        if progress_cb:
            progress_cb("voice_ready", i, len(_VOICE_PHRASES), phrase)
        time.sleep(0.8)   # prep time

        if progress_cb:
            progress_cb("voice_recording", i, len(_VOICE_PHRASES), phrase)

        audio = sd.rec(int(_DUR * _SR), samplerate=_SR,
                       channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        mfcc = _compute_mfcc(audio)
        all_mfccs.append(mfcc)
        logger.info(f"  Phrase {i+1} captured")

        if progress_cb:
            progress_cb("voice_done", i, len(_VOICE_PHRASES), phrase)
        time.sleep(0.4)

    mean_mfcc = np.mean(all_mfccs, axis=0)
    logger.info(f"Voice MFCC captured ({len(all_mfccs)} phrases)")
    return mean_mfcc


# ─────────────────────────────────────────────────────────────────────────────
#  SAVE TO CSV
# ─────────────────────────────────────────────────────────────────────────────
_DATA_DIR      = os.path.join(os.path.dirname(__file__), "..", "data")
_BIO_CSV       = os.path.join(_DATA_DIR, "biometrics.csv")
_BIO_FIELDS    = [
    "user_id",
    "enrolled_at",
    "face_embedding",    # 1404 floats, pipe-separated
    "face_ratios",       # 20 floats,   pipe-separated
    "hand_signature",    # 15 floats,   pipe-separated
    "voice_mfcc",        # 13 floats,   pipe-separated
    "voice_enabled",     # 1/0
]
_lock = threading.Lock()


def _arr_to_str(arr):
    return "|".join(f"{v:.6f}" for v in arr)


def _str_to_arr(s):
    return np.array([float(v) for v in s.split("|")], dtype=np.float32)


def save_biometrics(user_id: int,
                    face_embedding,
                    face_ratios,
                    hand_signature,
                    voice_mfcc) -> bool:
    """Write biometric vectors for user_id to biometrics.csv."""
    os.makedirs(_DATA_DIR, exist_ok=True)

    with _lock:
        # Ensure header exists
        if not os.path.exists(_BIO_CSV):
            with open(_BIO_CSV, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=_BIO_FIELDS).writeheader()

        # Read existing rows
        rows = []
        with open(_BIO_CSV, "r", newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f)
                    if r["user_id"] != str(user_id)]

        # Add / replace row for this user
        rows.append({
            "user_id":        str(user_id),
            "enrolled_at":    time.strftime("%Y-%m-%dT%H:%M:%S"),
            "face_embedding": _arr_to_str(face_embedding)
                              if face_embedding is not None else "",
            "face_ratios":    _arr_to_str(face_ratios)
                              if face_ratios is not None else "",
            "hand_signature": _arr_to_str(hand_signature)
                              if hand_signature is not None else "",
            "voice_mfcc":     _arr_to_str(voice_mfcc)
                              if voice_mfcc is not None else "",
            "voice_enabled":  "0" if voice_mfcc is None else "1",
        })

        with open(_BIO_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_BIO_FIELDS)
            w.writeheader()
            w.writerows(rows)

    logger.info(f"Biometrics saved for user_id={user_id}")
    return True


def load_biometrics(user_id: int):
    """
    Load stored biometric vectors for a user.
    Returns dict with numpy arrays, or None if not found.
    """
    if not os.path.exists(_BIO_CSV):
        return None

    with _lock:
        with open(_BIO_CSV, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["user_id"] == str(user_id):
                    return {
                        "face_embedding": _str_to_arr(row["face_embedding"])
                                          if row["face_embedding"] else None,
                        "face_ratios":    _str_to_arr(row["face_ratios"])
                                          if row["face_ratios"] else None,
                        "hand_signature": _str_to_arr(row["hand_signature"])
                                          if row["hand_signature"] else None,
                        "voice_mfcc":     _str_to_arr(row["voice_mfcc"])
                                          if row["voice_mfcc"] else None,
                        "voice_enabled":  row["voice_enabled"] == "1",
                        "enrolled_at":    row["enrolled_at"],
                    }
    return None
