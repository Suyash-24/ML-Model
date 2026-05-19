"""
auth.py  —  Eyecon Authentication & User Data Manager
────────────────────────────────────────────────────────
CSV-backed storage — zero lock issues, zero dependencies.

Storage files (auto-created in data/):
  • users.csv          — accounts & profiles
  • sessions.csv       — per-session analytics
  • commands.csv       — every command fired
  • calibration.csv    — eye calibration profiles

Validation rules (production-grade):
  Username : 3–20 chars · lowercase · letters/digits/underscores/periods
             must start with a letter · no consecutive special chars · unique
  Email    : valid format · lowercase · unique
  Password : 8+ chars · 1 uppercase · 1 lowercase · 1 digit · 1 special char
             cannot contain the username
"""

import csv
import hashlib
import secrets
import json
import os
import re
import threading
from datetime import datetime
from utils.logger import EyeconLogger

# ── Paths ────────────────────────────────────────────────────────────────────
_DATA_DIR       = os.path.join(os.path.dirname(__file__), "data")
_USERS_CSV      = os.path.join(_DATA_DIR, "users.csv")
_SESSIONS_CSV   = os.path.join(_DATA_DIR, "sessions.csv")
_COMMANDS_CSV   = os.path.join(_DATA_DIR, "commands.csv")
_CALIBRATION_CSV = os.path.join(_DATA_DIR, "calibration.csv")

# ── Thread safety ────────────────────────────────────────────────────────────
_lock = threading.Lock()

logger = EyeconLogger("Auth")

# ── CSV column definitions ───────────────────────────────────────────────────
_USER_FIELDS = [
    "id", "username", "email", "password_hash", "salt",
    "full_name", "age_group", "use_case", "disability", "country",
    "preferred_mode", "gaze_smooth_frames", "blink_click", "dwell_click",
    "gesture_sensitivity", "voice_enabled",
    "created_at", "last_login", "is_active",
]

_SESSION_FIELDS = [
    "id", "user_id", "started_at", "ended_at", "duration_secs",
    "eye_commands", "gesture_commands", "voice_commands", "total_commands",
    "avg_fps", "avg_latency_ms", "blink_clicks", "dwell_clicks",
    "gestures_fired", "voice_cmds_fired",
]

_COMMAND_FIELDS = [
    "id", "user_id", "session_id", "fired_at",
    "source", "command", "confidence", "latency_ms",
]

_CALIBRATION_FIELDS = [
    "id", "user_id", "created_at", "cal_data", "accuracy", "is_active",
]


# ─────────────────────────────────────────────────────────────────────────────
#  CSV HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_csv(path, fields):
    """Create CSV with header row if it doesn't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()


def _read_all(path, fields):
    """Read all rows from a CSV file. Returns list of dicts."""
    _ensure_csv(path, fields)
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_all(path, fields, rows):
    """Overwrite entire CSV with the given rows."""
    _ensure_csv(path, fields)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _append_row(path, fields, row):
    """Append a single row to a CSV file."""
    _ensure_csv(path, fields)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writerow(row)


def _next_id(rows):
    """Return the next auto-increment ID for the given rows."""
    if not rows:
        return 1
    return max(int(r.get("id", 0)) for r in rows) + 1


# ─────────────────────────────────────────────────────────────────────────────
#  DATABASE INIT
# ─────────────────────────────────────────────────────────────────────────────
_db_initialised = False


def init_db():
    """Ensure all CSV files exist with correct headers (runs once)."""
    global _db_initialised
    if _db_initialised:
        return
    _ensure_csv(_USERS_CSV, _USER_FIELDS)
    _ensure_csv(_SESSIONS_CSV, _SESSION_FIELDS)
    _ensure_csv(_COMMANDS_CSV, _COMMAND_FIELDS)
    _ensure_csv(_CALIBRATION_CSV, _CALIBRATION_FIELDS)
    _db_initialised = True
    logger.info("Database initialised (CSV)")


# ─────────────────────────────────────────────────────────────────────────────
#  PASSWORD HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _hash_password(password: str):
    """Return (hash, salt) tuple using SHA-256 + random salt."""
    salt    = secrets.token_hex(32)
    pw_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return pw_hash, salt


def _verify_password(password: str, pw_hash: str, salt: str) -> bool:
    return hashlib.sha256((password + salt).encode()).hexdigest() == pw_hash


# ─────────────────────────────────────────────────────────────────────────────
#  VALIDATION  (production-grade rules)
# ─────────────────────────────────────────────────────────────────────────────
_USERNAME_RE   = re.compile(r"^[a-z][a-z0-9._]{2,19}$")
_NO_CONSEC_RE  = re.compile(r"[._]{2}")
_EMAIL_RE      = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)
_SPECIAL_CHARS = set("!@#$%^&*()_+-=[]{}|;':\",./<>?`~")


def validate_username(username: str):
    """
    Rules:
      • 3–20 characters
      • Lowercase letters, digits, underscores, periods only
      • Must start with a letter
      • No consecutive periods or underscores  (e.g. user__1 or user..1)
      • Cannot end with a period or underscore
    Returns (clean_username, error_message_or_None)
    """
    raw = username.strip()
    clean = raw.lower()

    if len(clean) < 3:
        return clean, "Username must be at least 3 characters."
    if len(clean) > 20:
        return clean, "Username cannot exceed 20 characters."
    if not _USERNAME_RE.match(clean):
        if not clean[0].isalpha():
            return clean, "Username must start with a letter."
        return clean, "Username can only contain lowercase letters, digits, underscores, and periods."
    if _NO_CONSEC_RE.search(clean):
        return clean, "Username cannot have consecutive periods or underscores."
    if clean[-1] in "._":
        return clean, "Username cannot end with a period or underscore."
    return clean, None


def validate_email(email: str):
    """
    Rules:
      • Valid email format  (user@domain.tld)
      • Stored as lowercase
    Returns (clean_email, error_message_or_None)
    """
    clean = email.strip().lower()
    if not clean:
        return clean, "Email address is required."
    if not _EMAIL_RE.match(clean):
        return clean, "Please enter a valid email address."
    return clean, None


def validate_password(password: str, username: str = ""):
    """
    Rules:
      • Minimum 8 characters
      • At least 1 uppercase letter  (A–Z)
      • At least 1 lowercase letter  (a–z)
      • At least 1 digit             (0–9)
      • At least 1 special character  (!@#$%… etc.)
      • Cannot contain the username
    Returns error_message or None
    """
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if not re.search(r"[A-Z]", password):
        return "Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return "Password must contain at least one lowercase letter."
    if not re.search(r"\d", password):
        return "Password must contain at least one digit."
    if not any(ch in _SPECIAL_CHARS for ch in password):
        return "Password must contain at least one special character (!@#$%… etc.)."
    if username and username.lower() in password.lower():
        return "Password cannot contain your username."
    return None


def validate_full_name(name: str):
    """Optional field — if provided, 2–50 chars, letters/spaces/hyphens only."""
    clean = name.strip()
    if not clean:
        return clean, None                    # optional
    if len(clean) < 2:
        return clean, "Name must be at least 2 characters."
    if len(clean) > 50:
        return clean, "Name cannot exceed 50 characters."
    if not re.match(r"^[a-zA-Z\s\-'.]+$", clean):
        return clean, "Name can only contain letters, spaces, hyphens, and apostrophes."
    return clean, None


# ─────────────────────────────────────────────────────────────────────────────
#  USER MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
def register_user(username, email, password, full_name="",
                  age_group="", use_case="", disability="", country=""):
    """
    Register a new user with full validation.
    Returns: (True, user_dict) on success, (False, error_message) on failure.
    """
    init_db()

    # ── Validate username ─────────────────────────────────────────────────
    username, err = validate_username(username)
    if err:
        return False, err

    # ── Validate email ────────────────────────────────────────────────────
    email, err = validate_email(email)
    if err:
        return False, err

    # ── Validate password ─────────────────────────────────────────────────
    err = validate_password(password, username)
    if err:
        return False, err

    # ── Validate full name ────────────────────────────────────────────────
    full_name, err = validate_full_name(full_name)
    if err:
        return False, err

    pw_hash, salt = _hash_password(password)
    now = datetime.now().isoformat()

    with _lock:
        rows = _read_all(_USERS_CSV, _USER_FIELDS)

        # ── Uniqueness checks ────────────────────────────────────────────
        for row in rows:
            if row["username"].lower() == username:
                return False, "Username already taken. Try another one."
            if row["email"].lower() == email:
                return False, "An account with this email already exists."

        new_id = _next_id(rows)
        user_row = {
            "id":                   str(new_id),
            "username":             username,
            "email":                email,
            "password_hash":        pw_hash,
            "salt":                 salt,
            "full_name":            full_name,
            "age_group":            age_group,
            "use_case":             use_case,
            "disability":           disability,
            "country":              country,
            "preferred_mode":       "MULTI",
            "gaze_smooth_frames":   "6",
            "blink_click":          "1",
            "dwell_click":          "1",
            "gesture_sensitivity":  "1.0",
            "voice_enabled":        "1",
            "created_at":           now,
            "last_login":           "",
            "is_active":            "1",
        }
        _append_row(_USERS_CSV, _USER_FIELDS, user_row)

    logger.info(f"New user registered: {username} ({email})")

    # Return a safe user dict (no password hash)
    safe = dict(user_row)
    safe.pop("password_hash", None)
    safe.pop("salt", None)
    safe["id"] = new_id
    return True, safe


def login_user(username_or_email: str, password: str):
    """
    Verify credentials.
    Returns: (True, user_dict) on success, (False, error_message) on failure.
    """
    init_db()
    lookup = username_or_email.strip().lower()

    with _lock:
        rows = _read_all(_USERS_CSV, _USER_FIELDS)

        target = None
        for row in rows:
            if (row["username"].lower() == lookup or
                    row["email"].lower() == lookup):
                if row.get("is_active", "1") == "1":
                    target = row
                    break

        if not target:
            return False, "User not found."

        if not _verify_password(password, target["password_hash"], target["salt"]):
            return False, "Incorrect password."

        # Update last_login
        target["last_login"] = datetime.now().isoformat()
        _write_all(_USERS_CSV, _USER_FIELDS, rows)

    user = dict(target)
    user.pop("password_hash", None)
    user.pop("salt", None)
    user["id"] = int(user["id"])
    logger.info(f"User logged in: {target['username']}")
    return True, user


def get_user(user_id: int):
    with _lock:
        rows = _read_all(_USERS_CSV, _USER_FIELDS)
        for row in rows:
            if str(row.get("id")) == str(user_id):
                user = dict(row)
                user.pop("password_hash", None)
                user.pop("salt", None)
                return user
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  ACCOUNT EDITS
# ─────────────────────────────────────────────────────────────────────────────
def update_password(user_id: int, current: str, new: str):
    """Verify current password and replace it. Returns (ok, message)."""
    if not new or len(new) < 4:
        return False, "New password must be at least 4 characters."
    init_db()
    with _lock:
        rows = _read_all(_USERS_CSV, _USER_FIELDS)
        target = None
        for r in rows:
            if str(r.get("id")) == str(user_id):
                target = r; break
        if target is None:
            return False, "User not found."
        if not _verify_password(current, target["password_hash"], target["salt"]):
            return False, "Current password is incorrect."
        pw_hash, salt = _hash_password(new)
        target["password_hash"] = pw_hash
        target["salt"]          = salt
        _write_all(_USERS_CSV, _USER_FIELDS, rows)
    logger.info(f"Password updated for user_id {user_id}")
    return True, "Password updated."


def update_email(user_id: int, new_email: str):
    new_email = (new_email or "").strip().lower()
    if "@" not in new_email or "." not in new_email:
        return False, "Invalid email address."
    init_db()
    with _lock:
        rows = _read_all(_USERS_CSV, _USER_FIELDS)
        for r in rows:
            if (r.get("email", "").lower() == new_email and
                    str(r.get("id")) != str(user_id)):
                return False, "Email already in use."
        target = None
        for r in rows:
            if str(r.get("id")) == str(user_id):
                target = r; break
        if target is None:
            return False, "User not found."
        target["email"] = new_email
        _write_all(_USERS_CSV, _USER_FIELDS, rows)
    logger.info(f"Email updated for user_id {user_id}")
    return True, "Email updated."


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
def start_session(user_id: int) -> int:
    init_db()
    with _lock:
        rows = _read_all(_SESSIONS_CSV, _SESSION_FIELDS)
        new_id = _next_id(rows)
        _append_row(_SESSIONS_CSV, _SESSION_FIELDS, {
            "id":               str(new_id),
            "user_id":          str(user_id),
            "started_at":       datetime.now().isoformat(),
            "ended_at":         "",
            "duration_secs":    "0",
            "eye_commands":     "0",
            "gesture_commands": "0",
            "voice_commands":   "0",
            "total_commands":   "0",
            "avg_fps":          "0",
            "avg_latency_ms":   "0",
            "blink_clicks":     "0",
            "dwell_clicks":     "0",
            "gestures_fired":   "{}",
            "voice_cmds_fired": "{}",
        })
    return new_id


def end_session(session_id: int, stats: dict):
    """Save final session stats on logout/close."""
    with _lock:
        rows = _read_all(_SESSIONS_CSV, _SESSION_FIELDS)
        for row in rows:
            if str(row.get("id")) == str(session_id):
                row["ended_at"]         = datetime.now().isoformat()
                row["duration_secs"]    = str(stats.get("duration_secs", 0))
                row["eye_commands"]     = str(stats.get("eye_commands", 0))
                row["gesture_commands"] = str(stats.get("gesture_commands", 0))
                row["voice_commands"]   = str(stats.get("voice_commands", 0))
                row["total_commands"]   = str(stats.get("total_commands", 0))
                row["avg_fps"]          = str(stats.get("avg_fps", 0))
                row["avg_latency_ms"]   = str(stats.get("avg_latency_ms", 0))
                row["blink_clicks"]     = str(stats.get("blink_clicks", 0))
                row["dwell_clicks"]     = str(stats.get("dwell_clicks", 0))
                row["gestures_fired"]   = json.dumps(stats.get("gestures_fired", {}))
                row["voice_cmds_fired"] = json.dumps(stats.get("voice_cmds_fired", {}))
                break
        _write_all(_SESSIONS_CSV, _SESSION_FIELDS, rows)
    logger.info(f"Session {session_id} ended — {stats.get('total_commands', 0)} commands")


def log_command(user_id, session_id, source, command, confidence=None, latency_ms=None):
    with _lock:
        rows = _read_all(_COMMANDS_CSV, _COMMAND_FIELDS)
        new_id = _next_id(rows)
        _append_row(_COMMANDS_CSV, _COMMAND_FIELDS, {
            "id":           str(new_id),
            "user_id":      str(user_id),
            "session_id":   str(session_id),
            "fired_at":     datetime.now().isoformat(),
            "source":       source,
            "command":       command,
            "confidence":   str(confidence) if confidence is not None else "",
            "latency_ms":   str(latency_ms) if latency_ms is not None else "",
        })


# ─────────────────────────────────────────────────────────────────────────────
#  CALIBRATION PROFILE
# ─────────────────────────────────────────────────────────────────────────────
def save_calibration(user_id: int, cal_data: dict, accuracy: float = None):
    with _lock:
        rows = _read_all(_CALIBRATION_CSV, _CALIBRATION_FIELDS)
        # Deactivate old profiles for this user
        for row in rows:
            if str(row.get("user_id")) == str(user_id):
                row["is_active"] = "0"

        new_id = _next_id(rows)
        rows.append({
            "id":         str(new_id),
            "user_id":    str(user_id),
            "created_at": datetime.now().isoformat(),
            "cal_data":   json.dumps(cal_data),
            "accuracy":   str(accuracy) if accuracy is not None else "",
            "is_active":  "1",
        })
        _write_all(_CALIBRATION_CSV, _CALIBRATION_FIELDS, rows)


def load_calibration(user_id: int):
    with _lock:
        rows = _read_all(_CALIBRATION_CSV, _CALIBRATION_FIELDS)
        active = [
            r for r in rows
            if str(r.get("user_id")) == str(user_id)
            and r.get("is_active") == "1"
        ]
    if active:
        latest = max(active, key=lambda r: int(r["id"]))
        return json.loads(latest["cal_data"]), float(latest["accuracy"]) if latest["accuracy"] else None
    return None, None