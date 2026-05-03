"""
modules/voice_engine.py
───────────────────────
Voice Command System

Uses SpeechRecognition + Google Web Speech API (or offline Vosk).
Runs in a background thread, firing commands asynchronously.

Supported commands (extendable via config/commands.json):
  "open chrome"        → launch Chrome
  "scroll down/up"     → scroll page
  "increase volume"    → volume up
  "decrease volume"    → volume down
  "take screenshot"    → save screenshot
  "close window"       → Alt+F4
  "switch tab"         → Ctrl+Tab
  "stop eyecon"        → graceful shutdown
  "click"              → left click
  "right click"        → right click
  "press enter"        → Enter key
  "go back"            → Alt+Left
"""

import speech_recognition as sr
import pyautogui
import subprocess
import threading
import time
import json
import os
import platform
import webbrowser
from utils.logger import EyeconLogger


class VoiceEngine:
    def __init__(self, config, feedback):
        self.cfg      = config
        self.feedback = feedback
        self.logger   = EyeconLogger("VoiceEngine")

        self.recogniser   = sr.Recognizer()
        self.microphone_available = False

        try:
            self.microphone   = sr.Microphone()
            # Calibrate ambient noise once at startup
            with self.microphone as src:
                self.logger.info("Calibrating microphone to ambient noise…")
                self.recogniser.adjust_for_ambient_noise(src, duration=1.5)
            self.recogniser.energy_threshold = max(
                self.recogniser.energy_threshold,
                self.cfg.get("mic_energy_threshold", 300),
            )
            self.microphone_available = True
        except (OSError, AttributeError, Exception) as e:
            self.logger.warning(f"Microphone not available ({e}). Voice commands disabled.")
            self.microphone = None

        # Command registry: phrase → handler method name
        self._commands = self._load_commands()

        # State
        self.stop_requested = False
        self.paused         = False
        self.last_command   = None
        self.command_count  = 0
        self._lock          = threading.Lock()

        self.logger.info(f"VoiceEngine ready — {len(self._commands)} commands registered")

    # ─────────────────────────────────────────────────────────────────────
    #  COMMAND REGISTRY
    # ─────────────────────────────────────────────────────────────────────
    def _load_commands(self):
        """Load commands from JSON then merge built-ins."""
        path = os.path.join(os.path.dirname(__file__),
                            "..", "config", "commands.json")
        custom = {}
        if os.path.exists(path):
            with open(path) as f:
                custom = json.load(f)

        built_in = {
            # ── Navigation ─────────────────────────────────────
            "scroll down":        self._cmd_scroll_down,
            "scroll up":          self._cmd_scroll_up,
            "go back":            self._cmd_go_back,
            "go forward":         self._cmd_go_forward,
            "switch tab":         self._cmd_switch_tab,
            "close tab":          self._cmd_close_tab,
            "new tab":            self._cmd_new_tab,
            "close window":       self._cmd_close_window,
            # ── Mouse ──────────────────────────────────────────
            "click":              self._cmd_click,
            "right click":        self._cmd_right_click,
            "double click":       self._cmd_double_click,
            # ── Keyboard ───────────────────────────────────────
            "press enter":        self._cmd_enter,
            "press escape":       self._cmd_escape,
            "press space":        self._cmd_space,
            "copy":               self._cmd_copy,
            "paste":              self._cmd_paste,
            "undo":               self._cmd_undo,
            # ── System ─────────────────────────────────────────
            "increase volume":    self._cmd_vol_up,
            "decrease volume":    self._cmd_vol_down,
            "mute":               self._cmd_mute,
            "take screenshot":    self._cmd_screenshot,
            "open chrome":        self._cmd_open_chrome,
            "open file explorer": self._cmd_open_file_explorer,
            "open explorer":      self._cmd_open_file_explorer,
            "close chrome":       self._cmd_close_chrome,
            "quit chrome":        self._cmd_close_chrome,
            "open notepad":       self._cmd_open_notepad,
            "lock screen":        self._cmd_lock_screen,
            # ── Eyecon control ─────────────────────────────────
            "stop eyecon":        self._cmd_stop,
            "pause eyecon":       self._cmd_pause,
            "resume eyecon":      self._cmd_resume,
            "calibrate":          self._cmd_calibrate,
        }

        # Merge (custom override built-in if clashes)
        return {**built_in, **{k: self._make_custom_handler(v)
                                for k, v in custom.items()}}

    def _make_custom_handler(self, action_str):
        def handler():
            self.logger.info(f"Custom action: {action_str}")
            os.system(action_str)
        return handler

    # ─────────────────────────────────────────────────────────────────────
    #  LISTEN LOOP  (runs in background thread)
    # ─────────────────────────────────────────────────────────────────────
    def listen_loop(self):
        if not self.microphone_available:
            self.logger.warning("Voice listening loop disabled (microphone unavailable)")
            return

        self.logger.info("Voice listening loop started")
        while not self.stop_requested:
            if self.paused:
                time.sleep(0.2)
                continue
            try:
                with self.microphone as src:
                    audio = self.recogniser.listen(
                        src,
                        timeout=3,
                        phrase_time_limit=5,
                    )
                text = self.recogniser.recognize_google(audio).lower().strip()
                self.logger.info(f"Heard: '{text}'")
                self._dispatch(text)
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                self.logger.warning(f"Speech API error: {e}")
                time.sleep(2)
            except Exception as e:
                self.logger.error(f"Voice loop error: {e}")
                time.sleep(1)

    # ─────────────────────────────────────────────────────────────────────
    #  DISPATCH
    # ─────────────────────────────────────────────────────────────────────
    def _dispatch(self, text):
        """Find best matching command and execute it."""
        # Exact match first
        if text in self._commands:
            return self._run(text, self._commands[text])

        # Partial / fuzzy match
        best, best_score = None, 0
        for phrase in self._commands:
            score = self._similarity(text, phrase)
            if score > best_score and score >= 0.75:
                best, best_score = phrase, score

        if best:
            self.logger.info(f"Matched '{text}' → '{best}' (score={best_score:.2f})")
            return self._run(best, self._commands[best])

        self.logger.info(f"No command matched for: '{text}'")

    def _run(self, phrase, handler):
        with self._lock:
            self.last_command = phrase
            self.command_count += 1
        self.feedback.beep(880, 60)
        self.feedback.visual_flash("voice")
        threading.Thread(target=handler, daemon=True).start()

    def _similarity(self, a, b):
        """Token overlap score."""
        a_tok = set(a.split())
        b_tok = set(b.split())
        if not b_tok:
            return 0.0
        return len(a_tok & b_tok) / len(b_tok)

    # ─────────────────────────────────────────────────────────────────────
    #  COMMAND HANDLERS
    # ─────────────────────────────────────────────────────────────────────
    def _cmd_scroll_down(self):  pyautogui.scroll(-5)
    def _cmd_scroll_up(self):    pyautogui.scroll(5)
    def _cmd_go_back(self):      pyautogui.hotkey("alt", "left")
    def _cmd_go_forward(self):   pyautogui.hotkey("alt", "right")
    def _cmd_switch_tab(self):   pyautogui.hotkey("ctrl", "tab")
    def _cmd_close_tab(self):    pyautogui.hotkey("ctrl", "w")
    def _cmd_new_tab(self):      pyautogui.hotkey("ctrl", "t")
    def _cmd_close_window(self): pyautogui.hotkey("alt", "f4")
    def _cmd_click(self):        pyautogui.click()
    def _cmd_right_click(self):  pyautogui.rightClick()
    def _cmd_double_click(self): pyautogui.doubleClick()
    def _cmd_enter(self):        pyautogui.press("enter")
    def _cmd_escape(self):       pyautogui.press("escape")
    def _cmd_space(self):        pyautogui.press("space")
    def _cmd_copy(self):         pyautogui.hotkey("ctrl", "c")
    def _cmd_paste(self):        pyautogui.hotkey("ctrl", "v")
    def _cmd_undo(self):         pyautogui.hotkey("ctrl", "z")
    def _cmd_mute(self):         pyautogui.press("volumemute")

    def _cmd_vol_up(self):
        for _ in range(3): pyautogui.press("volumeup")

    def _cmd_vol_down(self):
        for _ in range(3): pyautogui.press("volumedown")

    def _cmd_screenshot(self):
        ts   = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(os.path.expanduser("~"), "Pictures",
                            f"eyecon_{ts}.png")
        pyautogui.screenshot(path)
        self.feedback.speak(f"Screenshot saved")
        self.logger.info(f"Screenshot → {path}")

    def _cmd_open_chrome(self):
        self.feedback.speak("Opening Chrome")
        system = platform.system()
        if system == "Windows":
            chrome_paths = [
                os.path.join(os.environ.get("PROGRAMFILES", ""),
                             "Google", "Chrome", "Application", "chrome.exe"),
                os.path.join(os.environ.get("PROGRAMFILES(X86)", ""),
                             "Google", "Chrome", "Application", "chrome.exe"),
                os.path.join(os.environ.get("LOCALAPPDATA", ""),
                             "Google", "Chrome", "Application", "chrome.exe"),
            ]
            for path in chrome_paths:
                if path and os.path.isfile(path):
                    subprocess.Popen([path])
                    return
            for cmd in ("chrome", "msedge"):
                try:
                    subprocess.Popen([cmd])
                    return
                except Exception:
                    continue
            webbrowser.open("about:blank")
            self.logger.warning("Chrome not found; opened default browser instead")
            return

        if system == "Darwin":
            try:
                subprocess.Popen(["open", "-a", "Google Chrome"])
                return
            except Exception:
                webbrowser.open("about:blank")
                self.logger.warning("Chrome not found; opened default browser instead")
                return

        for cmd in ("google-chrome", "google-chrome-stable", "chromium-browser", "chromium"):
            try:
                subprocess.Popen([cmd])
                return
            except Exception:
                continue
        webbrowser.open("about:blank")
        self.logger.warning("Chrome not found; opened default browser instead")

    def _cmd_open_notepad(self):
        self.feedback.speak("Opening Notepad")
        try:    subprocess.Popen(["notepad"])
        except: subprocess.Popen(["gedit"])

    def _cmd_open_file_explorer(self):
        self.feedback.speak("Opening File Explorer")
        system = platform.system()
        home = os.path.expanduser("~")
        if system == "Windows":
            try:
                os.startfile(home)
                return
            except Exception:
                try:
                    subprocess.Popen(["explorer", home])
                    return
                except Exception:
                    pass
        elif system == "Darwin":
            try:
                subprocess.Popen(["open", home])
                return
            except Exception:
                pass
        else:
            for cmd in ("xdg-open", "nautilus", "gio"):
                try:
                    if cmd == "gio":
                        subprocess.Popen(["gio", "open", home])
                    else:
                        subprocess.Popen([cmd, home])
                    return
                except Exception:
                    continue
        self.logger.warning("File Explorer launch failed")

    def _cmd_close_chrome(self):
        self.feedback.speak("Closing Chrome")
        system = platform.system()
        if system == "Windows":
            try:
                subprocess.Popen(["taskkill", "/IM", "chrome.exe", "/F"])
                return
            except Exception:
                pass
        elif system == "Darwin":
            try:
                subprocess.Popen(["osascript", "-e", 'quit app "Google Chrome"'])
                return
            except Exception:
                pass
        else:
            for cmd in ("pkill",):
                try:
                    subprocess.Popen([cmd, "-f", "chrome"])
                    return
                except Exception:
                    continue
        self.logger.warning("Chrome close failed or Chrome not running")

    def _cmd_lock_screen(self):
        self.feedback.speak("Locking screen")
        if platform.system() == "Windows":
            import ctypes; ctypes.windll.user32.LockWorkStation()
        else:
            os.system("loginctl lock-session")

    def _cmd_stop(self):
        self.feedback.speak("Stopping Eyecon. Goodbye.")
        self.stop_requested = True

    def _cmd_pause(self):
        self.paused = True
        self.feedback.speak("Eyecon paused")

    def _cmd_resume(self):
        self.paused = False
        self.feedback.speak("Eyecon resumed")

    def _cmd_calibrate(self):
        self.feedback.speak("Starting eye calibration")
        # Signal main loop to run calibration
        self._calibrate_requested = True

    # ─────────────────────────────────────────────────────────────────────
    #  PUBLIC CONTROL
    # ─────────────────────────────────────────────────────────────────────
    def stop(self):
        self.stop_requested = True

    def get_last_command(self):
        with self._lock:
            cmd = self.last_command
            self.last_command = None
        return cmd
