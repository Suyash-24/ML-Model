"""
modules/wednesday_engine.py  —  WEDNESDAY AI Engine (Jarvis-level)
Wake word: "wednesday" via Whisper Tiny (free, offline, no API key)
"""
import os, json, time, threading, subprocess, webbrowser, shutil, math, re, queue
import urllib.request
import urllib.error
import numpy as np
from datetime import datetime, timedelta
from utils.logger import EyeconLogger

logger = EyeconLogger("WEDNESDAY")

try:
    import whisper as _whisper; _WHISPER_OK = True
except ImportError:
    _WHISPER_OK = False; logger.warning("whisper not installed")

try:
    import sounddevice as sd; _SD_OK = True
except ImportError:
    _SD_OK = False; logger.warning("sounddevice not installed")

try:
    import ollama as _ollama; _OLLAMA_OK = True
except ImportError:
    _OLLAMA_OK = False; logger.warning("ollama not installed")

try:
    import pyautogui; pyautogui.FAILSAFE = False; _GUI_OK = True
except ImportError:
    _GUI_OK = False

try:
    import pyttsx3; _PYTTSX3_OK = True
except ImportError:
    _PYTTSX3_OK = False

try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs import play as _eleven_play; _ELEVEN_OK = True
except ImportError:
    _ELEVEN_OK = False

try:
    import pyperclip; _CLIP_OK = True
except ImportError:
    _CLIP_OK = False

try:
    import ctypes
    from ctypes import wintypes
    _WIN_IDLE_OK = True
except Exception:
    _WIN_IDLE_OK = False

if _WIN_IDLE_OK:
    class _LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

# ── Fast-path keyword intents (no LLM needed) ────────────────────────────────
# Each entry: ([keyword_phrases], action, params, speech)
_FAST_INTENTS = [
    (["youtube","your tube","you tube","utube","open tube",
      "open view","open the view","you view","u tube","utube",
      "tube","open you","view tube","open yt","openview"],
     "open_url", {"url":"https://www.youtube.com"}, "Opening YouTube, Boss."),
    (["netflix"],
     "open_url", {"url":"https://www.netflix.com"}, "Opening Netflix, Boss."),
    (["github"],
     "open_url", {"url":"https://www.github.com"}, "Opening GitHub, Boss."),
    (["gmail"],
     "open_url", {"url":"https://mail.google.com"}, "Opening Gmail, Boss."),
    (["google"],
     "open_url", {"url":"https://www.google.com"}, "Opening Google, Boss."),
    (["vs code","vscode","visual studio code","b.s. code","ves code","face code"],
     "_vscode", {}, "Launching VS Code, Boss."),
    (["open chrome","launch chrome","start chrome"],
     "open_app", {"app":"chrome"}, "Opening Chrome, Boss."),
    (["open discord","launch discord","start discord"],
     "open_app", {"app":"discord"}, "Opening Discord, Boss."),
    (["open spotify","launch spotify","start spotify"],
     "open_app", {"app":"spotify"}, "Opening Spotify, Boss."),
    (["open notepad","launch notepad"],
     "open_app", {"app":"notepad"}, "Opening Notepad, Boss."),
    (["open calculator","launch calculator","open calc"],
     "open_app", {"app":"calculator"}, "Opening Calculator, Boss."),
    (["open task manager","task manager"],
     "open_app", {"app":"task_manager"}, "Opening Task Manager, Boss."),
        (["stop listening","stop wake","stop wake word","go idle",
            "pause listening","sleep listening"],
         "stop_listening", {}, "Listening paused."),
        (["open file explorer","open file exploror","open explorer",
            "file explorer","file exploror","file manager"],
         "open_app", {"app":"explorer"}, "Opening File Explorer, Boss."),
    (["volume up","louder","turn up","increase volume"],
     "volume", {"direction":"up","steps":3}, "Volume up."),
    (["volume down","quieter","turn down","decrease volume"],
     "volume", {"direction":"down","steps":3}, "Volume down."),
    (["mute"],
     "mute", {}, "Muted."),
    (["play","pause","play pause","play music","pause music"],
     "media_play_pause", {}, "Done."),
    (["next song","next track","skip song"],
     "media_next", {}, "Skipping to next."),
    (["previous song","prev song","prev track"],
     "media_prev", {}, "Going back."),
    (["lock screen","lock computer","lock pc"],
     "lock_screen", {}, "Locking your screen, Boss."),
    (["screenshot","take screenshot","screen shot"],
     "screenshot", {}, "Screenshot taken."),
    (["what time","current time","tell me the time"],
     "tell_time", {}, ""),
    (["what date","today's date","what day is it"],
     "tell_date", {}, ""),
    (["shut down","shutdown","turn off"],
     "shutdown", {"delay":0}, "Shutting down, Boss."),
    (["restart","reboot"],
     "restart", {"delay":0}, "Restarting, Boss."),
    (["open downloads","show downloads"],
     "open_folder", {"folder":"downloads"}, "Opening Downloads."),
    (["open documents","show documents"],
     "open_folder", {"folder":"documents"}, "Opening Documents."),
    (["open desktop","show desktop"],
     "open_folder", {"folder":"desktop"}, "Opening Desktop."),
]

_WAKE_VARIANTS = {
    # Correct spelling
    "wednesday",
    # Common Whisper mishearings of 'wednesday'
    "wendsday", "wensday", "wednsday", "wenesday",
    "wednesdays", "wensddays", "wednessday", "wednesay",
    "wend", "wendys", "wendie",
    "windsday", "windsdey", "winzday",
    # Whisper tiny phonetic near-misses observed in testing
    "wensde", "wenz", "wented", "weddins", "weddin",
    "wedn", "wensd", "wednsd",
}

_SYSTEM_PROMPT = """You are WEDNESDAY, the central AI of the user's operating system.
You receive voice commands transcribed by Whisper STT — which is IMPERFECT. You MUST correct obvious transcription errors and deduce the real intent.

COMMON WHISPER TRANSCRIPTION ERRORS (always correct these):
- "open your tube" / "open you tube" / "open utube" / "open tube" → user means "open YouTube"
- "b.s. code" / "be es code" / "face code" / "ves code" / "vs cord" → user means "open VS Code"
- "open google chrome" → open_app with app: "chrome"
- "open discord" / "open this cord" → open_app with app: "discord"
- "volume up" / "turn it up" / "louder" → volume up
- "what time" / "what's the time" → tell_time
General rule: use context and phonetics to deduce intent. Assume something sensible even with garbled input.

You have two modes:
1. SYSTEM CONTROLLER — control the PC (apps, websites, media, files, system)
2. CONVERSATIONAL AI — answer questions, write code, chat

ALWAYS respond with ONE valid JSON object ONLY. No markdown, no explanation outside the JSON.
{"action": "<name>", "params": {}, "confidence": 0.9, "speech": "..."}

AVAILABLE ACTIONS:
- chat: {} → user is conversing, asking a question, requesting info or code
- open_url: {"url": "https://..."} → open ANY website directly
- web_search: {"query": "...", "engine": "google|youtube|bing"} → search the web
- open_app: {"app": "<name>"} → open a desktop app by name (chrome, notepad, vs_code, discord, spotify, calculator)
 - open_app: {"app": "<name>"} → open a desktop app by name (chrome, notepad, vs_code, discord, spotify, calculator, explorer)
- run_command: {"cmd": "...", "shell": true} → run a shell/terminal command
- media_play_pause / media_next / media_prev / media_stop: {}
- volume: {"direction": "up|down", "steps": 3}
- shutdown: {"delay": 0} / restart: {"delay": 0} / sleep: {} / lock_screen: {}
- open_file: {"path": "..."} / open_folder: {"folder": "downloads|documents|desktop|pictures|music"}
- type_text: {"text": "...", "press_enter": false}
- tell_time / tell_date / calculate: {"expression": "..."} / tell_weather: {"city": "..."}
- eye_enable: {"enabled": true|false} / gesture_enable: {"enabled": true|false}
- stop_listening: {} → stop continuous listening (clap twice to resume)
- clarify: {} → ONLY use this if input is truly random gibberish with zero guessable intent

DECISION EXAMPLES:
- "open youtube" / "open your tube" → {"action":"open_url","params":{"url":"https://www.youtube.com"},"confidence":0.97,"speech":"Opening YouTube, Boss."}
- "open vs code" / "open b.s. code" → {"action":"open_app","params":{"app":"vs_code"},"confidence":0.97,"speech":"Launching VS Code, Boss."}
- "open file explorer" / "open explorer" → {"action":"open_app","params":{"app":"explorer"},"confidence":0.97,"speech":"Opening File Explorer, Boss."}
- "open chrome" → {"action":"open_app","params":{"app":"chrome"},"confidence":0.97,"speech":"Opening Chrome, Boss."}
- "volume up" → {"action":"volume","params":{"direction":"up","steps":3},"confidence":0.99,"speech":"Turning up the volume."}
- "what time is it" → {"action":"tell_time","params":{},"confidence":0.99,"speech":"Let me check the time."}

RULES:
- Confidence threshold to execute is 0.45 — do NOT use clarify for anything above this
- For commands: keep speech SHORT (1 sentence, e.g. "On it, Boss.")
- For chat/questions: spoken answer only, max 2 sentences, max 30 words
- NEVER write code, markdown, or bullet lists in speech — this is read aloud by TTS
- If asked to write code or give a detailed answer, say: "Please type that in the chat for a full answer."
- NEVER leave speech empty
- The wake word is "wednesday" — strip it from your understanding
- MULTI-STEP REQUESTS: pick the FIRST action only. Never output more than ONE JSON object.
"""

_CHAT_SYSTEM_PROMPT = """You are Wednesday, a helpful AI assistant embedded in a desktop app.
Answer questions fully and helpfully. Write code when asked, explain concepts, be conversational.
You are running in a chat panel — the user reads your responses, they are NOT spoken aloud.
If web search results are provided, use them and cite sources with URLs when relevant.
If asked about the current date, day, or time, use the local time provided in system context.
You may use markdown formatting: **bold**, `code`, ```code blocks```, bullet lists, etc.
Be thorough and precise. Do NOT output JSON. Respond in plain readable text.
"""


class WednesdayEngine:
    WAKE_WHISPER_MODEL = "base"   # Use base for both wake + STT (tiny too inaccurate)
    STT_WHISPER_MODEL  = "base"
    LLM_MODEL          = "llama3.2:3b"
    SAMPLE_RATE        = 16000
    WAKE_CHUNK_SECS    = 2.5
    CMD_MAX_SECS       = 5
    FOLLOWUP_WINDOW_SECS = 7
    FOLLOWUP_MAX_TURNS   = 2
    IDLE_STOP_SECS       = 300

    # Common app name → executable mapping
    APP_MAP = {
        "chrome": ["chrome", "google-chrome"],
        "youtube": ["_URL_https://www.youtube.com"],
        "gmail":   ["_URL_https://mail.google.com"],
        "github":  ["_URL_https://github.com"],
        "twitter": ["_URL_https://x.com"],
        "reddit":  ["_URL_https://www.reddit.com"],
        "netflix": ["_URL_https://www.netflix.com"],
        "firefox": ["firefox"],
        "edge": ["msedge"],
        "notepad": ["notepad"],
        "calculator": ["calc"],
        "explorer": ["explorer"],
        "files": ["explorer"],
        "file_explorer": ["explorer"],
        "fileexplorer": ["explorer"],
        "settings": ["ms-settings:"],
        "task_manager": ["taskmgr"],
        "vs_code":            ["_VSCODE_"],
        "vscode":             ["_VSCODE_"],
        "visual_studio_code": ["_VSCODE_"],
        "vs":                 ["_VSCODE_"],
        "code":               ["_VSCODE_"],
        "discord": ["discord"],
        "spotify": ["spotify"],
        "steam": ["steam"],
        "obs": ["obs64", "obs"],
        "vlc": ["vlc"],
        "paint": ["mspaint"],
        "word": ["winword"],
        "excel": ["excel"],
        "powerpoint": ["powerpnt"],
        "cmd": ["cmd"],
        "powershell": ["powershell"],
        "terminal": ["wt", "powershell"],
        "whatsapp": ["whatsapp"],
        "zoom": ["zoom"],
        "teams": ["teams"],
        "snipping": ["snippingtool"],
        "clock": ["ms-clock:"],
        "photos": ["ms-photos:"],
        "mail": ["outlook", "thunderbird"],
    }

    FOLDER_MAP = {
        "downloads": os.path.expanduser("~/Downloads"),
        "documents": os.path.expanduser("~/Documents"),
        "desktop":   os.path.expanduser("~/Desktop"),
        "pictures":  os.path.expanduser("~/Pictures"),
        "music":     os.path.expanduser("~/Music"),
        "videos":    os.path.expanduser("~/Videos"),
        "home":      os.path.expanduser("~"),
    }

    SEARCH_URLS = {
        "google":  "https://www.google.com/search?q=",
        "youtube": "https://www.youtube.com/results?search_query=",
        "bing":    "https://www.bing.com/search?q=",
    }

    def __init__(self, config, feedback):
        self.cfg      = config
        self.feedback = feedback
        self.stop_requested = False
        self.paused         = False
        self._last_command  = None
        self._last_speech   = None
        self._lock          = threading.Lock()
        self._in_turn       = False
        self._timers        = []

        self._memory_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "wednesday_memory.json")
        self._memory_max  = config.get("wednesday_memory_turns", 10)
        self._history     = self._load_memory()   # voice command history
        self._chat_history: list = []             # text chat history (separate)

        self._sphere_cb = None
        self._chat_cb   = None
        self._page_cb   = None
        self._username  = config.get("username", "Boss")
        self._sphere_hold_until = 0.0
        self._sphere_idle_timer = None
        self._sphere_lock = threading.Lock()
        self._noise_floor = 0.01
        self._listen_always_on = bool(config.get("voice_listen_always_on", True))
        self._idle_stop_secs = int(config.get("voice_idle_stop_secs", self.IDLE_STOP_SECS))
        self._vad_block_ms = int(config.get("voice_vad_block_ms", 40))
        self._vad_speech_mult = float(config.get("voice_vad_speech_mult", 2.2))
        self._vad_silence_mult = float(config.get("voice_vad_silence_mult", 1.2))
        self._vad_hangover_frames = int(config.get("voice_vad_hangover_frames", 8))
        self._vad_min_ms = int(config.get("voice_vad_min_ms", 400))
        self._vad_max_secs = int(config.get("voice_vad_max_secs", 6))
        self._listening_active = False
        self._audio_q = queue.Queue(maxsize=60)
        self._stt_queue = queue.Queue()
        self._audio_thread = None
        self._cmd_queue = queue.Queue()
        self._stt_worker = threading.Thread(target=self._stt_worker_loop, daemon=True)
        self._stt_worker.start()
        self._cmd_worker = threading.Thread(target=self._command_worker, daemon=True)
        self._cmd_worker.start()

        self._serper_key = (
            config.get("serper_api_key", "") or os.getenv("SERPER_API_KEY", "")
        ).strip()
        self._serper_auto = bool(config.get("serper_auto_search", False))
        self._serper_max_results = max(1, min(10, int(config.get("serper_max_results", 5))))
        self._serper_cache_ttl = max(60, int(config.get("serper_cache_ttl_secs", 900)))
        self._serper_cache = {}
        self._serper_lock = threading.Lock()
        self._wake_model = None
        self._stt_model  = None
        self._speaking   = False   # True while TTS is playing

        # TTS
        self._tts   = None
        self._eleven = None
        if _PYTTSX3_OK:
            try:
                self._tts = pyttsx3.init()
                voices = self._tts.getProperty("voices")
                zira = next((v for v in voices if "zira" in v.name.lower()), None)
                if zira:
                    self._tts.setProperty("voice", zira.id)
                self._tts.setProperty("rate", 162)
                self._tts.setProperty("volume", 0.93)
                logger.info("TTS: Microsoft Zira (free)")
            except Exception as e:
                logger.warning(f"pyttsx3 failed: {e}"); self._tts = None

        api_key = config.get("elevenlabs_api_key", "")
        if _ELEVEN_OK and api_key and api_key not in ("", "YOUR_ELEVENLABS_KEY"):
            try:
                self._eleven = ElevenLabs(api_key=api_key)
                logger.info("TTS: ElevenLabs premium")
            except Exception: pass

        logger.info("WednesdayEngine ready — say 'wednesday' to activate")

        # Ensure Ollama is running before anything else
        threading.Thread(target=self._ensure_ollama, daemon=True).start()
        # Pre-load STT model in background so first voice command doesn't stall
        threading.Thread(target=self._preload_stt, daemon=True).start()
        # Start audio stream for clap + continuous VAD
        self._ensure_audio_stream()
        if self._listen_always_on:
            self._start_listening()

    def _ensure_ollama(self):
        """Check if Ollama is reachable; if not, start it automatically."""
        import urllib.request, urllib.error
        url = "http://127.0.0.1:11434"

        def _reachable():
            try:
                urllib.request.urlopen(url, timeout=2)
                return True
            except Exception:
                return False

        if _reachable():
            logger.info("Ollama already running.")
            return

        logger.info("Ollama not running — attempting to start…")

        # Find ollama executable
        ollama_exe = shutil.which("ollama")
        if not ollama_exe:
            for tmpl in [
                r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe",
                r"C:\Program Files\Ollama\ollama.exe",
            ]:
                p = os.path.expandvars(tmpl)
                if os.path.isfile(p):
                    ollama_exe = p
                    break

        if not ollama_exe:
            logger.error("ollama executable not found — install from https://ollama.com/download")
            self._chat("system", "Ollama not found. Install it from ollama.com/download")
            return

        try:
            # Start in background, detached from this process
            subprocess.Popen(
                [ollama_exe, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
            logger.info(f"Started ollama serve ({ollama_exe})")
        except Exception as e:
            logger.error(f"Failed to start Ollama: {e}")
            return

        # Wait up to 12 seconds for Ollama to become reachable
        for attempt in range(12):
            time.sleep(1)
            if _reachable():
                logger.info(f"Ollama is up (took ~{attempt + 1}s)")
                return
        logger.warning("Ollama started but didn't respond within 12s — it may still be loading")

    def _preload_stt(self):
        """Load Whisper base model in background so it's ready for first voice command."""
        try:
            if not _WHISPER_OK: return
            logger.info("Pre-loading Whisper Base STT model…")
            dev = "cuda" if self._cuda_ok() else "cpu"
            self._stt_model = _whisper.load_model(self.STT_WHISPER_MODEL, device=dev)
            logger.info(f"Whisper Base STT ready on {dev}")
        except Exception as e:
            logger.warning(f"STT pre-load failed: {e}")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def set_sphere_callback(self, fn):
        self._sphere_cb = fn
        if self._listening_active:
            self._sphere("listening")
    def set_chat_callback(self, fn):   self._chat_cb   = fn
    def set_page_callback(self, fn):   self._page_cb   = fn
    def set_history_callback(self, fn):
        """Receives every dispatched action: fn(entry: dict)."""
        self._history_cb = fn

    def _switch_page(self, name: str):
        if self._page_cb:
            try: self._page_cb(name)
            except: pass

    def _emit_history(self, action: str, params: dict, ok: bool, error: str = ""):
        if getattr(self, "_history_cb", None):
            try:
                self._history_cb({
                    "t": time.strftime("%H:%M:%S"),
                    "action": action,
                    "params": params or {},
                    "ok": ok,
                    "error": error,
                })
            except: pass

    def _sphere(self, s):
        if not self._sphere_cb:
            return
        with self._sphere_lock:
            if s != "idle" and self._sphere_idle_timer:
                try:
                    self._sphere_idle_timer.cancel()
                except Exception:
                    pass
                self._sphere_idle_timer = None

            if s == "listening":
                self._sphere_hold_until = time.time() + 0.4

            if s == "idle" and time.time() < self._sphere_hold_until:
                delay = max(0.0, self._sphere_hold_until - time.time())
                if self._sphere_idle_timer:
                    try:
                        self._sphere_idle_timer.cancel()
                    except Exception:
                        pass
                self._sphere_idle_timer = threading.Timer(
                    delay, lambda: self._sphere("idle")
                )
                self._sphere_idle_timer.daemon = True
                self._sphere_idle_timer.start()
                return

        try:
            self._sphere_cb(s)
        except Exception:
            pass

    def _chat(self, role, text):
        if self._chat_cb:
            try: self._chat_cb(role, text)
            except: pass

    def _start_listening(self):
        if self._listening_active:
            return
        self._set_listening_active(True)
        self._ensure_audio_stream()
        self._sphere("listening")

    def _stop_listening(self):
        if not self._listening_active:
            return
        self._set_listening_active(False)
        self._sphere("idle")

    def _ensure_audio_stream(self):
        if not _SD_OK:
            return
        if self._audio_thread and self._audio_thread.is_alive():
            return
        self._audio_thread = threading.Thread(
            target=self._audio_stream_loop, daemon=True)
        self._audio_thread.start()

    def _set_listening_active(self, active: bool):
        self._listening_active = bool(active)

    def _get_idle_secs(self):
        if not _WIN_IDLE_OK:
            return None
        try:
            lii = _LASTINPUTINFO()
            lii.cbSize = ctypes.sizeof(_LASTINPUTINFO)
            if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
                return None
            tick = ctypes.windll.kernel32.GetTickCount()
            return max(0, int(tick - lii.dwTime)) / 1000.0
        except Exception:
            return None

    def _should_stop_listening(self):
        if not self._listening_active:
            return False
        idle = self._get_idle_secs()
        if idle is not None and idle >= self._idle_stop_secs:
            logger.info(f"Listening stopped due to idle ({idle:.0f}s)")
            return True
        return False

    def _enqueue_command(self, text: str, source: str = "voice"):
        t = (text or "").strip()
        if len(t) < 3:
            return
        self._cmd_queue.put((t, source))

    def _command_worker(self):
        while not self.stop_requested:
            try:
                item = self._cmd_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            text, source = item
            try:
                result = self._fast_match(text)
                if result:
                    logger.info(f"Fast match → {result['action']}")
                else:
                    if not self._listening_active:
                        self._sphere("processing")
                    result = self._query_llm(text)
                if result is None:
                    self._speak("Couldn't process that, Boss.")
                else:
                    self._dispatch(result, text, source=source)
            finally:
                self._cmd_queue.task_done()
                if self._listening_active:
                    self._sphere("listening")

    def _audio_stream_loop(self):
        if not _SD_OK:
            return
        block = int(self.SAMPLE_RATE * (self._vad_block_ms / 1000.0))
        if block <= 0:
            block = int(self.SAMPLE_RATE * 0.04)
        block_secs = block / float(self.SAMPLE_RATE)
        min_frames = max(1, int(self._vad_min_ms / (block_secs * 1000.0)))
        max_frames = max(1, int(self._vad_max_secs / max(block_secs, 0.001)))
        pre_roll_frames = max(1, int(0.2 / max(block_secs, 0.001)))
        hangover = max(1, self._vad_hangover_frames)

        from collections import deque
        pre_roll = deque(maxlen=pre_roll_frames)
        in_speech = False
        speech_buf = []
        speech_frames = 0
        silence_frames = 0

        ambient = self._noise_floor
        first_clap_time = None
        in_clap = False
        min_gap = 0.15
        max_gap = 1.2

        def _cb(indata, frames, t, status):
            try:
                self._audio_q.put_nowait(indata.copy().flatten())
            except queue.Full:
                pass

        try:
            stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=block,
                callback=_cb,
            )
            stream.start()
        except Exception as e:
            logger.error(f"Audio stream open failed: {e}")
            return

        try:
            while not self.stop_requested:
                try:
                    chunk = self._audio_q.get(timeout=0.5)
                except queue.Empty:
                    if self._listening_active and self._should_stop_listening():
                        self._stop_listening()
                    continue

                rms = float(np.sqrt(np.mean(chunk ** 2)))

                if not in_speech:
                    self._noise_floor = (self._noise_floor * 0.98) + (rms * 0.02)

                speech_gate = max(0.009, self._noise_floor * self._vad_speech_mult)
                silence_gate = max(0.006, self._noise_floor * self._vad_silence_mult)

                if not self._listening_active and in_speech:
                    in_speech = False
                    speech_buf = []
                    speech_frames = 0
                    silence_frames = 0
                    pre_roll.clear()

                if self._listening_active:
                    if self._should_stop_listening():
                        self._stop_listening()
                        in_speech = False
                        speech_buf = []
                        speech_frames = 0
                        silence_frames = 0
                        pre_roll.clear()
                        continue

                    if not in_speech:
                        pre_roll.append(chunk)
                        if rms >= speech_gate:
                            in_speech = True
                            speech_buf = list(pre_roll)
                            speech_frames = len(speech_buf)
                            silence_frames = 0
                    else:
                        speech_buf.append(chunk)
                        speech_frames += 1
                        if rms < silence_gate:
                            silence_frames += 1
                        else:
                            silence_frames = 0
                        if silence_frames >= hangover or speech_frames >= max_frames:
                            if speech_frames >= min_frames:
                                audio = np.concatenate(speech_buf)
                                self._stt_queue.put(audio)
                            in_speech = False
                            speech_buf = []
                            speech_frames = 0
                            silence_frames = 0
                            pre_roll.clear()
                    continue

                # Not listening — clap detection only
                ambient = (ambient * 0.98) + (rms * 0.02)
                threshold = max(0.04, min(0.20, ambient * 6))
                decay = max(0.015, ambient * 2.5)
                rising = rms > threshold and not in_clap
                if rms > threshold:
                    in_clap = True
                elif rms < decay:
                    in_clap = False

                if rising:
                    now = time.time()
                    if first_clap_time is None:
                        first_clap_time = now
                        logger.info(f"Clap 1 ✓  RMS={rms:.4f}  thresh={threshold:.4f}")
                    else:
                        gap = now - first_clap_time
                        if min_gap <= gap <= max_gap:
                            logger.info(f"Clap 2 ✓  gap={gap:.2f}s — ACTIVATING")
                            first_clap_time = None
                            in_clap = False
                            self._start_listening()
                        elif gap > max_gap:
                            first_clap_time = now
                            logger.info(f"Clap 1 reset  RMS={rms:.4f}")

                if first_clap_time and (time.time() - first_clap_time) > max_gap + 0.3:
                    first_clap_time = None
                    in_clap = False
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def _split_commands(self, text: str) -> list:
        cleaned = (text or "").strip()
        lower = cleaned.lower()
        for w in _WAKE_VARIANTS:
            if lower.startswith(w):
                cleaned = cleaned[len(w):].strip(", ")
                lower = cleaned.lower()
                break
        parts = re.split(r"\b(?:and then|then|and)\b", cleaned, flags=re.IGNORECASE)
        return [p.strip(" ,.") for p in parts if len(p.strip()) >= 3]

    def _stt_worker_loop(self):
        while not self.stop_requested:
            try:
                audio = self._stt_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if audio is None:
                break
            try:
                text = self._transcribe_full(audio)
                if not text or len(text.strip()) < 3:
                    continue
                cleaned = text.strip()
                lower = cleaned.lower()
                tokens = re.findall(r"[a-z]+", lower)
                single_ok = {
                    "mute", "pause", "play", "stop", "lock", "shutdown", "restart",
                    "sleep", "screenshot", "volume", "google", "youtube", "gmail",
                    "github", "discord", "spotify", "chrome", "notepad", "calculator",
                    "explorer", "downloads", "documents", "desktop",
                }
                if len(tokens) <= 1 and (not tokens or tokens[0] not in single_ok):
                    continue
                for part in self._split_commands(cleaned):
                    self._enqueue_command(part, "voice")
            finally:
                self._stt_queue.task_done()

    # ── Fast-path matcher ──────────────────────────────────────────────────────────
    def _fast_match(self, text: str):
        """Return a pre-built result dict if text matches a known keyword intent.
        Returns None if no fast match — caller should fall back to LLM."""
        t = text.lower().strip()
        if re.search(r"\b(open|launch|start)\b.*\bexplor", t):
            return {"action": "open_app", "params": {"app": "explorer"},
                    "confidence": 0.99, "speech": "Opening File Explorer, Boss."}
        for keywords, action, params, speech in _FAST_INTENTS:
            if any(kw in t for kw in keywords):
                return {"action": action, "params": params,
                        "confidence": 0.99, "speech": speech}
        return None

    # ── Wake word loop ─────────────────────────────────────────────────────────
    def start_wake_word_loop(self):
        """Start audio stream for clap + continuous listening."""
        self._ensure_audio_stream()
        logger.info("Wake loop started — clap twice to activate Wednesday")

    def _clap_loop(self):
        """Detect two claps within 1.2s using sd.InputStream (zero-gap continuous monitoring)."""
        if not _SD_OK:
            logger.warning("sounddevice not available — clap wake disabled")
            return

        import queue as _queue
        BLOCK   = int(self.SAMPLE_RATE * 0.04)   # 40ms blocks
        MIN_GAP = 0.15
        MAX_GAP = 1.2
        rms_q   = _queue.Queue()

        def _cb(indata, frames, t, status):
            rms_q.put(float(np.sqrt(np.mean(indata ** 2))))

        try:
            stream = sd.InputStream(samplerate=self.SAMPLE_RATE, channels=1,
                                    dtype="float32", blocksize=BLOCK, callback=_cb)
            stream.start()
        except Exception as e:
            logger.error(f"Clap stream open failed: {e}"); return

        # ── Adaptive calibration: 0.5s of ambient ──────────────────────────
        try:
            logger.info("Calibrating mic for clap detection…")
            samples = [rms_q.get(timeout=1.0) for _ in range(12)]
            ambient = float(np.mean(samples))
            THRESHOLD = max(0.04, min(0.20, ambient * 6))
            DECAY     = max(0.015, ambient * 2.5)
            self._noise_floor = max(0.005, ambient)
            logger.info(f"Clap calibration OK — ambient={ambient:.4f}  "
                        f"threshold={THRESHOLD:.4f}  decay={DECAY:.4f}")
            logger.info("WEDNESDAY ready — clap twice to give a voice command")
        except Exception as e:
            logger.error(f"Clap calibration failed: {e}"); stream.stop(); return

        first_clap_time = None
        in_clap         = False

        try:
            while not self.stop_requested:
                if self._speaking or self._in_turn or self.paused:
                    # Drain queue so stale RMS values don't trigger after resume
                    while not rms_q.empty():
                        try: rms_q.get_nowait()
                        except _queue.Empty: break
                    first_clap_time = None; in_clap = False
                    time.sleep(0.05); continue

                try:
                    rms = rms_q.get(timeout=0.2)
                except _queue.Empty:
                    continue

                rising = rms > THRESHOLD and not in_clap
                if rms > THRESHOLD:  in_clap = True
                elif rms < DECAY:    in_clap = False

                if rising:
                    now = time.time()
                    if first_clap_time is None:
                        first_clap_time = now
                        logger.info(f"Clap 1 ✓  RMS={rms:.4f}  thresh={THRESHOLD:.4f}")
                    else:
                        gap = now - first_clap_time
                        if MIN_GAP <= gap <= MAX_GAP:
                            logger.info(f"Clap 2 ✓  gap={gap:.2f}s — ACTIVATING")
                            first_clap_time = None; in_clap = False
                            self._set_listening_active(True)
                            if not self._in_turn:
                                self._sphere("listening")
                                threading.Thread(target=self._handle_voice_turn,
                                                 args=(True,), daemon=True).start()
                        elif gap > MAX_GAP:
                            first_clap_time = now
                            logger.info(f"Clap 1 reset  RMS={rms:.4f}")

                if first_clap_time and (time.time() - first_clap_time) > MAX_GAP + 0.3:
                    first_clap_time = None; in_clap = False

        finally:
            stream.stop(); stream.close()

    def _wake_loop(self):
        if not _WHISPER_OK or not _SD_OK:
            missing = []
            if not _WHISPER_OK: missing.append("openai-whisper")
            if not _SD_OK:      missing.append("sounddevice")
            msg = (f"Wake word disabled — missing: {', '.join(missing)}. "
                   f"Install with: pip install {' '.join(missing)}")
            logger.warning(msg)
            self._chat("system", msg)
            return

        # Check microphone availability up front
        try:
            mics = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
            if not mics:
                msg = "Wake word disabled — no input microphone detected."
                logger.error(msg); self._chat("system", msg); return
            logger.info(f"Microphone OK — {len(mics)} input device(s) available")
        except Exception as e:
            logger.warning(f"Could not query audio devices: {e}")

        # Wait for pre-loaded base model (up to 30s) — avoids loading it twice
        logger.info("Waiting for Whisper base model to be ready…")
        for _ in range(60):
            if self._stt_model is not None:
                break
            time.sleep(0.5)

        if self._stt_model is None:
            # Pre-load didn't finish — load it now
            try:
                logger.info("Loading Whisper base for wake word…")
                dev = "cuda" if self._cuda_ok() else "cpu"
                self._stt_model = _whisper.load_model(self.WAKE_WHISPER_MODEL, device=dev)
            except Exception as e:
                logger.error(f"Whisper load failed: {e}")
                self._chat("system", f"Wake word disabled — Whisper load failed: {e}")
                return

        self._wake_model = self._stt_model   # share the same model instance
        logger.info("WEDNESDAY ready — whisper-base listening for 'wednesday'")

        chunk = int(self.SAMPLE_RATE * self.WAKE_CHUNK_SECS)

        # Quick mic test — verify audio is captured
        try:
            test = sd.rec(int(self.SAMPLE_RATE * 0.5), samplerate=self.SAMPLE_RATE,
                          channels=1, dtype="float32")
            sd.wait()
            rms = float(np.sqrt(np.mean(test ** 2)))
            logger.info(f"Mic test: RMS={rms:.5f} ({'OK — audio detected' if rms > 0.001 else 'WARNING — very quiet, check mic'})")
        except Exception as e:
            logger.error(f"Mic test failed: {e}")

        # Overlapped recording: record NEXT chunk while transcribing CURRENT one
        # This eliminates the ~1-2s dead gap during transcription
        prev_audio = None
        while not self.stop_requested:
            if self.paused or self._in_turn:
                time.sleep(0.2); prev_audio = None; continue
            try:
                # Start recording this chunk
                audio = sd.rec(chunk, samplerate=self.SAMPLE_RATE, channels=1, dtype="float32")

                # While recording, transcribe the PREVIOUS chunk (if any)
                if prev_audio is not None:
                    self._process_wake_chunk(prev_audio)

                sd.wait()  # wait for current recording to finish
                prev_audio = audio.flatten()

            except Exception as e:
                logger.error(f"Wake loop: {e}"); time.sleep(1); prev_audio = None

    def _process_wake_chunk(self, audio_flat):
        """Transcribe a wake-word audio chunk and react if detected."""
        rms = float(np.sqrt(np.mean(audio_flat ** 2)))
        if rms < 0.003:
            return  # silence — skip transcription entirely (saves CPU)
        model = self._stt_model or self._wake_model
        if model is None:
            return
        res  = model.transcribe(audio_flat, fp16=False, language="en",
                                without_timestamps=True,
                                condition_on_previous_text=False)
        text = res.get("text", "").lower().strip()
        if not text or len(text) < 2:
            return
        # Log what Whisper hears (diagnostic — helps debug wake word)
        logger.info(f"Wake heard: '{text}' (RMS={rms:.4f})")
        words = set(text.replace(",","").replace(".","").replace("'","").split())
        # Direct word match
        detected = bool(words & _WAKE_VARIANTS)
        # Substring match for phonetic near-misses
        if not detected:
            detected = any(v in text for v in _WAKE_VARIANTS)
        if detected:
            logger.info(f"Wake word DETECTED in: '{text}'")
            # Extract any command that followed the wake word in THIS chunk
            inline_cmd = text
            for w in sorted(_WAKE_VARIANTS, key=len, reverse=True):
                idx = inline_cmd.find(w)
                if idx != -1:
                    inline_cmd = inline_cmd[idx + len(w):].strip(" ,.").strip()
                    break
            if len(inline_cmd) >= 4:
                logger.info(f"Inline command: '{inline_cmd}'")
                threading.Thread(
                    target=self._run_voice_command,
                    args=(inline_cmd,), daemon=True).start()
            else:
                # Just the wake word — listen for the command now
                self._sphere("listening")
                self._handle_voice_turn(False)

    # ── Voice pipeline ─────────────────────────────────────────────────────────────
    def _handle_voice_turn(self, continuous: bool = False):
        """Record a single voice command; continuous mode enables always-on listening."""
        if continuous:
            self._start_listening()
            return
        if self._in_turn: return
        self._in_turn = True
        try:
            self._sphere("listening")
            audio = self._record_audio(self.CMD_MAX_SECS)
            if audio is None:
                return
            if not self._listening_active:
                self._sphere("processing")
            transcript = self._transcribe_full(audio)
            if not transcript or len(transcript.strip()) < 3:
                return
            # Strip wake word if it leaked into the recording
            cleaned = transcript.strip()
            for w in _WAKE_VARIANTS:
                if cleaned.lower().startswith(w):
                    cleaned = cleaned[len(w):].strip(", "); break
            if cleaned:
                self._run_voice_command(cleaned)
        finally:
            if not self._listening_active:
                self._sphere("idle")
            self._in_turn = False

    def _run_voice_command(self, text: str):
        """Queue a transcribed voice command for sequential execution."""
        self._enqueue_command(text, "voice")

    def _dispatch(self, result: dict, original: str, source: str = "voice"):
        action     = result.get("action",     "chat")
        params     = result.get("params",     {})
        speech     = result.get("speech",     "")
        confidence = result.get("confidence", 0.0)
        logger.info(f"LLM → {action} ({confidence:.2f})")
        if not speech:
            if action == "clarify": speech = "Could you clarify that, Boss?"
            elif action == "chat":  speech = "I'm not quite sure what to do with that, Boss. Let's try again."
            else:                   speech = "On it, Boss."
            
        if speech:
            if not self._listening_active:
                self._sphere("speaking")
            self._speak(speech)   # voice-only: no chat callback
        if action not in ("chat", "clarify", "tell_time", "tell_date", "calculate") and confidence >= 0.45:
            self._execute(action, params)
            with self._lock: self._last_command = action
        elif action == "tell_time":
            msg = f"It's {datetime.now().strftime('%I:%M %p')}, Boss."
            self._speak(msg)
        elif action == "tell_date":
            msg = f"Today is {datetime.now().strftime('%A, %B %d %Y')}."
            self._speak(msg)
        elif action == "calculate":
            self._do_calculate(params.get("expression",""), source=source)
        self._add_to_memory(original, speech)

    # ── Execute ────────────────────────────────────────────────────────────────
    def _execute(self, action: str, params: dict):
        try:
            # ── BROWSER ──
            if action == "open_url":
                url = params.get("url","")
                if not url.startswith("http"): url = "https://" + url
                webbrowser.open(url)

            elif action == "web_search":
                q   = params.get("query","")
                eng = params.get("engine","google").lower()
                url = self.SEARCH_URLS.get(eng, self.SEARCH_URLS["google"]) + q.replace(" ","+")
                webbrowser.open(url)

            elif action == "new_tab":       self._hk("ctrl","t")
            elif action == "close_tab":     self._hk("ctrl","w")
            elif action == "go_back":       self._hk("alt","left")
            elif action == "go_forward":    self._hk("alt","right")
            elif action == "refresh":       pyautogui.press("f5")
            elif action == "fullscreen":    pyautogui.press("f11")
            elif action == "zoom_in":       self._hk("ctrl","=")
            elif action == "zoom_out":      self._hk("ctrl","-")

            # ── APPS ──
            elif action == "_vscode":
                self._launch_vscode()

            elif action == "open_app":
                app = params.get("app","").lower().replace(" ","_").replace("-","_")
                cmds = self.APP_MAP.get(app, [app])
                # Handle URL pseudo-apps
                url_cmd = next((c for c in cmds if isinstance(c,str) and c.startswith("_URL_")), None)
                if url_cmd:
                    webbrowser.open(url_cmd[5:])
                elif cmds == ["_VSCODE_"] or app in ("vs_code","vscode","visual_studio_code","vs","code"):
                    self._launch_vscode()
                else:
                    self._launch(cmds)

            elif action == "close_window":   self._hk("alt","f4")
            elif action == "minimize_window":self._hk("win","down")
            elif action == "maximize_window":self._hk("win","up")
            elif action == "new_window":     self._hk("ctrl","n")
            elif action == "show_desktop":   self._hk("win","d")
            elif action == "switch_app":
                if params.get("direction","next")=="prev": self._hk("alt","shift","tab")
                else: self._hk("alt","tab")

            # ── MEDIA ──
            elif action == "media_play_pause": pyautogui.press("playpause")
            elif action == "media_next":       pyautogui.press("nexttrack")
            elif action == "media_prev":       pyautogui.press("prevtrack")
            elif action == "media_stop":       pyautogui.press("stop")
            elif action == "mute":             pyautogui.press("volumemute")
            elif action == "volume":
                key   = "volumeup" if params.get("direction")=="up" else "volumedown"
                steps = int(params.get("steps",3))
                for _ in range(steps): pyautogui.press(key)

            # ── SYSTEM ──
            elif action == "shutdown":
                d = int(params.get("delay",0))
                subprocess.Popen(f"shutdown /s /t {d}", shell=True)
            elif action == "restart":
                d = int(params.get("delay",0))
                subprocess.Popen(f"shutdown /r /t {d}", shell=True)
            elif action == "sleep":
                subprocess.Popen("rundll32.exe powrprof.dll,SetSuspendState 0,1,0", shell=True)
            elif action == "lock_screen":      self._hk("win","l")
            elif action == "task_manager":     self._hk("ctrl","shift","esc")
            elif action == "brightness":
                key   = "brightnessup" if params.get("direction")=="up" else "brightnessdown"
                for _ in range(int(params.get("steps",3))): pyautogui.press(key)
            elif action == "kill_process":
                name = params.get("name","")
                if name: subprocess.Popen(f"taskkill /f /im {name}", shell=True)
            elif action == "run_command":
                cmd = params.get("cmd","")
                if cmd: subprocess.Popen(cmd, shell=True)

            # ── FILES ──
            elif action == "open_file":
                path = params.get("path","")
                if path: os.startfile(path)
            elif action == "open_folder":
                folder = params.get("folder","home").lower()
                path   = self.FOLDER_MAP.get(folder, os.path.expanduser("~"))
                subprocess.Popen(["explorer", path])
            elif action == "create_folder":
                path = params.get("path","")
                if path: os.makedirs(path, exist_ok=True)
            elif action == "tell_weather":
                city = params.get("city","").replace(" ","+")
                webbrowser.open(f"https://wttr.in/{city}")

            # ── MOUSE & KEYBOARD ──
            elif action == "click":
                x, y = params.get("x"), params.get("y")
                btn  = params.get("button","left")
                if x and y: pyautogui.moveTo(x, y, duration=0.2)
                if btn=="double":  pyautogui.doubleClick()
                elif btn=="right": pyautogui.rightClick()
                else:              pyautogui.click()
            elif action == "scroll":
                d = params.get("direction","down")
                a = int(params.get("amount",3))
                if d in ("up","down"):
                    pyautogui.scroll(a if d=="up" else -a)
                else:
                    pyautogui.hscroll(a if d=="right" else -a)
            elif action == "move_mouse":
                pyautogui.moveTo(params.get("x",500), params.get("y",500), duration=0.3)
            elif action == "hotkey":
                keys = params.get("keys",[])
                if keys: pyautogui.hotkey(*keys)
            elif action == "type_text":
                text = params.get("text","")
                pyautogui.typewrite(text, interval=0.04)
                if params.get("press_enter"): pyautogui.press("enter")
            elif action == "select_all":  self._hk("ctrl","a")
            elif action == "copy":        self._hk("ctrl","c")
            elif action == "paste":       self._hk("ctrl","v")
            elif action == "undo":        self._hk("ctrl","z")
            elif action == "redo":        self._hk("ctrl","y")
            elif action == "save":        self._hk("ctrl","s")
            elif action == "find":
                self._hk("ctrl","f")
                txt = params.get("text","")
                if txt: pyautogui.typewrite(txt, interval=0.05)
            elif action == "screenshot":
                ts   = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.expanduser(f"~/Pictures/wednesday_{ts}.png")
                pyautogui.screenshot(path)
                logger.info(f"Screenshot: {path}")

            # ── CLIPBOARD ──
            elif action == "get_clipboard":
                if _CLIP_OK:
                    txt = pyperclip.paste()
                    self._chat("wednesday", f"Clipboard: {txt[:200]}")
                    self._speak(f"Clipboard contains: {txt[:100]}")
            elif action == "set_clipboard":
                if _CLIP_OK: pyperclip.copy(params.get("text",""))

            # ── EYECON ──
            elif action == "pause_system":   self.paused = True
            elif action == "resume_system":
                self.paused = False
                self._switch_page("Eye Control")
            elif action == "eye_enable":
                if params.get("enabled", True):
                    self._switch_page("Eye Control")
            elif action == "calibrate_eye":
                self._switch_page("Eye Control")
            elif action == "stop_listening":
                self._stop_listening()

            logger.info(f"Executed: {action}")
            self._emit_history(action, params, True)
        except Exception as e:
            logger.error(f"Execute error ({action}): {e}")
            self._emit_history(action, params, False, str(e))
            self._speak(f"Sorry Boss, I couldn't complete that: {e}")

    def _launch_vscode(self):
        """Find and launch VS Code on Windows via multiple strategies."""
        candidates = []
        # 1. PATH lookup (works when installer added code.cmd to PATH)
        w = shutil.which("code") or shutil.which("code.cmd")
        if w: candidates.append(w)
        # 2. Common install paths
        for tmpl in [
            r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe",
            r"%ProgramFiles%\Microsoft VS Code\Code.exe",
            r"%ProgramW6432%\Microsoft VS Code\Code.exe",
            r"C:\Program Files\Microsoft VS Code\Code.exe",
            r"C:\Program Files (x86)\Microsoft VS Code\Code.exe",
        ]:
            p = os.path.expandvars(tmpl)
            if os.path.isfile(p): candidates.append(p)
        for path in candidates:
            try:
                subprocess.Popen([path])
                logger.info(f"VS Code launched: {path}")
                return
            except Exception as e:
                logger.warning(f"VS Code attempt failed ({path}): {e}")
        # 3. Shell fallback — relies on PATHEXT resolution
        try:
            subprocess.Popen("code", shell=True)
            logger.info("VS Code launched via shell fallback")
            return
        except Exception as e:
            logger.error(f"VS Code shell launch failed: {e}")
            raise RuntimeError("VS Code not found. Install VS Code or add 'code' to PATH.") from e

    def _launch(self, cmds: list):
        """Try each command in list until one succeeds.
        Falls back to shell=True so Windows can resolve `.cmd`/`.bat`
        wrappers like `code.cmd`, `npm.cmd`, etc. from PATH.
        Raises the last error if every attempt fails — so the outer
        `_execute` can mark the action as failed in history.
        """
        last_err = None
        for cmd in cmds:
            # Expand environment variables in string commands
            if isinstance(cmd, str):
                cmd = os.path.expandvars(cmd)

            # ms-* protocol handlers go through os.startfile
            if isinstance(cmd, str) and cmd.startswith("ms-"):
                try:
                    os.startfile(cmd); return
                except Exception as e:
                    last_err = e; continue

            # Direct Popen first (fastest, exact behaviour)
            try:
                subprocess.Popen(cmd if isinstance(cmd, list) else [cmd])
                return
            except FileNotFoundError as e:
                last_err = e
            except Exception as e:
                last_err = e

            # Shell fallback — resolves PATHEXT (.cmd, .bat, .ps1)
            try:
                if isinstance(cmd, list):
                    cmd_str = subprocess.list2cmdline(cmd)
                else:
                    cmd_str = str(cmd)
                subprocess.Popen(cmd_str, shell=True)
                return
            except Exception as e:
                last_err = e

            # Last resort: os.startfile (user installs in %LOCALAPPDATA%)
            try:
                target = cmd[0] if isinstance(cmd, list) else cmd
                os.startfile(target)
                return
            except Exception as e:
                last_err = e

        if last_err is not None:
            raise last_err

    def _hk(self, *keys):
        pyautogui.hotkey(*keys)

    def _do_calculate(self, expr: str, source: str = "voice"):
        try:
            safe = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
            result = eval(expr, {"__builtins__": {}}, safe)
            msg = f"{expr} = {result}"
            self._chat("wednesday", msg)
            if source == "voice": self._speak(f"The answer is {result}")
        except Exception:
            if source == "voice": self._speak("I couldn't evaluate that expression, Boss.")
            self._chat("wednesday", "Calculation failed.")

    # ── Audio / STT ───────────────────────────────────────────────────────────
    def _record_audio(self, max_secs=5, prefix=None):
        """Record audio with simple VAD: stop early after short silence."""
        if not _SD_OK: return None
        try:
            chunk_secs  = 0.4
            chunk_size  = int(self.SAMPLE_RATE * chunk_secs)
            silence_rms = max(0.006, self._noise_floor * 0.8)
            speech_gate = max(0.009, self._noise_floor * 1.4)
            max_silence = 8       # consecutive silent chunks before stopping (~3.2s)
            max_chunks  = int(max_secs / chunk_secs)
            min_chunks  = 4       # always record at least ~1.6s

            chunks = []
            silent_streak = 0
            speech_seen = False
            if prefix is not None and len(prefix):
                p = np.array(prefix, dtype="float32").flatten()
                chunks.append(p)
                rms0 = float(np.sqrt(np.mean(p ** 2)))
                if rms0 < silence_rms:
                    silent_streak = 1
                if rms0 > speech_gate:
                    speech_seen = True

            remaining = max(0, max_chunks - len(chunks))
            if remaining == 0:
                return np.concatenate(chunks) if chunks else None

            for _ in range(remaining):
                c = sd.rec(chunk_size, samplerate=self.SAMPLE_RATE, channels=1, dtype="float32")
                sd.wait()
                chunks.append(c.flatten())
                rms = float(np.sqrt(np.mean(c ** 2)))
                if rms > speech_gate:
                    speech_seen = True
                if rms < silence_rms:
                    silent_streak += 1
                    if silent_streak >= max_silence and len(chunks) >= min_chunks:
                        break
                else:
                    silent_streak = 0

            if not speech_seen:
                return None
            return np.concatenate(chunks) if chunks else None
        except Exception as e:
            logger.error(f"Record: {e}"); return None

    def _wait_for_followup_chunk(self, wait_secs: float):
        """Listen briefly for speech onset and return the trigger chunk."""
        if not _SD_OK: return None
        try:
            chunk_secs = 0.2
            chunk_size = int(self.SAMPLE_RATE * chunk_secs)
            threshold  = max(0.010, self._noise_floor * 1.5)
            end = time.time() + max(0.0, wait_secs)
            hits = 0
            while time.time() < end and not self.stop_requested:
                c = sd.rec(chunk_size, samplerate=self.SAMPLE_RATE,
                           channels=1, dtype="float32")
                sd.wait()
                rms = float(np.sqrt(np.mean(c ** 2)))
                if rms > threshold:
                    hits += 1
                else:
                    hits = max(0, hits - 1)
                if hits >= 2:
                    return c.flatten()
            return None
        except Exception as e:
            logger.error(f"Follow-up listen: {e}")
            return None

    def _transcribe_full(self, audio):
        if not _WHISPER_OK: return ""
        if self._stt_model is None:
            logger.info("Loading Whisper Base…")
            self._stt_model = _whisper.load_model(
                self.STT_WHISPER_MODEL, device="cuda" if self._cuda_ok() else "cpu")
        try:
            r = self._stt_model.transcribe(audio, fp16=False, language="en",
                                           condition_on_previous_text=False)
            return r.get("text","").strip()
        except Exception as e:
            logger.error(f"STT: {e}"); return ""

    def _cuda_ok(self):
        try:
            import torch; return torch.cuda.is_available()
        except: return False

    # ── LLM ───────────────────────────────────────────────────────────────────
    @staticmethod
    def _extract_first_json(s: str) -> str:
        """Extract the first complete JSON object from a string."""
        depth, start = 0, None
        for i, c in enumerate(s):
            if c == '{':
                if start is None: start = i
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    return s[start:i + 1]
        return s

    def _should_search(self, text: str) -> bool:
        if not (self._serper_key and self._serper_auto):
            return False
        t = text.strip().lower()
        if len(t) < 4:
            return False
        if t in ("hi", "hello", "hey", "thanks", "thank you"):
            return False
        if any(term in t for term in (
            "holiday", "observance", "national day", "international day",
            "today in history", "news", "events"
        )):
            return True
        if any(term in t for term in (
            "what day", "what date", "today's date", "current date",
            "current day", "day today", "date today", "what time",
            "time is it", "current time"
        )):
            return False
        if any(term in t for term in (
            "tomorrow", "yesterday", "day after tomorrow", "day before yesterday"
        )):
            return False
        return True

    def _extract_date_from_text(self, text: str) -> datetime | None:
        t = (text or "").lower()

        # YYYY-MM-DD or YYYY/MM/DD
        m = re.search(r"(?<!\d)(\d{4})[/-](\d{1,2})[/-](\d{1,2})(?!\d)", t)
        if m:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                return datetime(y, mo, d)
            except ValueError:
                return None

        # DD/MM/YYYY or MM/DD/YYYY
        m = re.search(r"(?<!\d)(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?!\d)", t)
        if m:
            a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if a > 12 and b <= 12:
                d, mo = a, b
            elif b > 12 and a <= 12:
                mo, d = a, b
            else:
                d, mo = a, b
            try:
                return datetime(y, mo, d)
            except ValueError:
                return None

        months = {
            "jan": 1, "january": 1,
            "feb": 2, "february": 2,
            "mar": 3, "march": 3,
            "apr": 4, "april": 4,
            "may": 5,
            "jun": 6, "june": 6,
            "jul": 7, "july": 7,
            "aug": 8, "august": 8,
            "sep": 9, "sept": 9, "september": 9,
            "oct": 10, "october": 10,
            "nov": 11, "november": 11,
            "dec": 12, "december": 12,
        }
        month_re = (
            r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
            r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
            r"nov(?:ember)?|dec(?:ember)?)"
        )

        # 24 May 2026
        m = re.search(rf"(?<!\d)(\d{{1,2}})(?:st|nd|rd|th)?\s+{month_re}\s*,?\s*(\d{{4}})", t)
        if m:
            d = int(m.group(1))
            mo = months.get(m.group(2)[:3], 0)
            y = int(m.group(3))
            try:
                return datetime(y, mo, d)
            except ValueError:
                return None

        # May 24, 2026
        m = re.search(rf"{month_re}\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s*(\d{{4}})", t)
        if m:
            mo = months.get(m.group(1)[:3], 0)
            d = int(m.group(2))
            y = int(m.group(3))
            try:
                return datetime(y, mo, d)
            except ValueError:
                return None

        return None

    def _find_recent_date(self, text: str) -> datetime | None:
        direct = self._extract_date_from_text(text)
        if direct:
            return direct
        today = datetime.now().date()
        candidates = []
        for turn in reversed(self._chat_history[-12:]):
            for key in ("user", "assistant"):
                dt = self._extract_date_from_text(turn.get(key, ""))
                if dt:
                    candidates.append(dt)
            if candidates:
                break
        if not candidates:
            return None
        for dt in candidates:
            if dt.date() != today:
                return dt
        return candidates[0]

    def _local_date_answer(self, text: str) -> str | None:
        t = " ".join((text or "").lower().split())
        now = datetime.now()

        if "day after tomorrow" in t:
            dt = now + timedelta(days=2)
            return f"The day after tomorrow is {dt.strftime('%A, %B %d, %Y')}."
        if "day before yesterday" in t:
            dt = now - timedelta(days=2)
            return f"The day before yesterday was {dt.strftime('%A, %B %d, %Y')}."
        if "tomorrow" in t:
            dt = now + timedelta(days=1)
            return f"Tomorrow is {dt.strftime('%A, %B %d, %Y')}."
        if "yesterday" in t:
            dt = now - timedelta(days=1)
            return f"Yesterday was {dt.strftime('%A, %B %d, %Y')}."

        if any(term in t for term in ("that date", "this date", "that day")):
            dt = self._find_recent_date(text)
            if dt:
                return f"That date is {dt.strftime('%A, %B %d, %Y')}."
            return "Which date do you mean?"

        dt = self._extract_date_from_text(text)
        if dt:
            return f"{dt.strftime('%A, %B %d, %Y')}."

        if any(term in t for term in (
            "what time", "time is it", "current time", "time now"
        )):
            return f"It is {now.strftime('%I:%M %p')}."

        if any(term in t for term in (
            "what day", "what date", "today's date", "current date",
            "current day", "day today", "date today", "day is it"
        )):
            return f"Today is {now.strftime('%A, %B %d, %Y')}."

        return None

    def _serper_search(self, query: str) -> list:
        if not self._serper_key:
            return []
        q = (query or "").strip()
        if q.lower().startswith("search:"):
            q = q.split(":", 1)[1].strip()
        if not q:
            return []

        cache_key = q.lower()
        with self._serper_lock:
            cached = self._serper_cache.get(cache_key)
            if cached and (time.time() - cached["t"]) < self._serper_cache_ttl:
                return cached["items"]

        payload = json.dumps({"q": q}).encode("utf-8")
        req = urllib.request.Request(
            "https://google.serper.dev/search",
            data=payload,
            headers={
                "X-API-KEY": self._serper_key,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.warning(f"Serper search failed: {e}")
            return []

        items = []
        for it in data.get("organic", [])[: self._serper_max_results]:
            items.append({
                "title": it.get("title", ""),
                "link": it.get("link", ""),
                "snippet": it.get("snippet", ""),
            })

        if items:
            with self._serper_lock:
                self._serper_cache[cache_key] = {"t": time.time(), "items": items}
        return items

    def _format_search_results(self, items: list) -> str:
        lines = ["Web search results (Serper):"]
        for idx, it in enumerate(items, 1):
            title = (it.get("title") or "").strip()
            link = (it.get("link") or "").strip()
            snippet = (it.get("snippet") or "").strip().replace("\n", " ")
            if len(snippet) > 240:
                snippet = snippet[:237] + "..."
            if title or link:
                lines.append(f"{idx}. {title} — {link}")
            if snippet:
                lines.append(f"   {snippet}")
        return "\n".join(lines)

    def _query_chat_llm(self, text: str) -> str:
        """Plain-text ChatGPT-like response for the chat panel. No JSON, no TTS."""
        if not _OLLAMA_OK:
            return "Ollama is not installed. The chat AI is offline."
        local_answer = self._local_date_answer(text)
        if local_answer:
            return local_answer
        messages = [{"role": "system", "content": _CHAT_SYSTEM_PROMPT}]
        now = datetime.now()
        tz = time.tzname[0] if time.tzname else ""
        local_time = now.strftime("%Y-%m-%d %H:%M:%S")
        local_date = now.strftime("%B %d, %Y")
        local_day = now.strftime("%A")
        messages.append({
            "role": "system",
            "content": (
                f"Local time: {local_time} {tz}. "
                f"Local date: {local_date}. Local day: {local_day}."
            ),
        })
        if self._should_search(text):
            results = self._serper_search(text)
            if results:
                messages.append({
                    "role": "system",
                    "content": self._format_search_results(results),
                })
        for t in self._chat_history[-self._memory_max:]:
            messages += [{"role": "user", "content": t["user"]},
                         {"role": "assistant", "content": t["assistant"]}]
        messages.append({"role": "user", "content": text})
        try:
            resp = _ollama.chat(model=self.LLM_MODEL, messages=messages,
                                options={"temperature": 0.7, "num_predict": 1024,
                                         "num_ctx": 4096})
            answer = resp["message"]["content"].strip()
            self._chat_history.append({"user": text, "assistant": answer})
            if len(self._chat_history) > self._memory_max * 2:
                self._chat_history = self._chat_history[-self._memory_max:]
            return answer
        except Exception as e:
            logger.error(f"Chat LLM: {e}")
            return "Sorry, I couldn't reach the AI. Is Ollama running?"

    def _query_llm(self, text: str):
        if not _OLLAMA_OK:
            return {"action":"chat","params":{},"confidence":1.0,
                    "speech":"LLM offline. Install Ollama."}
        messages = [{"role":"system","content":_SYSTEM_PROMPT}]
        for t in self._history[-self._memory_max:]:
            messages += [{"role":"user","content":t["user"]},
                         {"role":"assistant","content":t["wednesday"]}]
        messages.append({"role":"user","content":text})
        try:
            resp = _ollama.chat(model=self.LLM_MODEL, messages=messages,
                                options={"temperature": 0.2, "num_predict": 400, "num_ctx": 2048})
            raw  = resp["message"]["content"].strip()
            logger.info(f"Raw LLM: {raw[:200]}")
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"): raw = raw[4:]
            # Extract only the FIRST JSON object (ignore multi-JSON output)
            raw = self._extract_first_json(raw.strip())
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("JSON parse failed — extracting speech from raw")
            import re
            # Attempt 2: try to re-extract the first JSON block and retry parse
            try:
                return json.loads(self._extract_first_json(raw.strip()))
            except Exception:
                pass
            # Attempt 3: regex-extract just the speech string value
            match = re.search(r'"speech"\s*:\s*"((?:[^"\\]|\\.)*?)"', raw, re.DOTALL)
            if match:
                speech = match.group(1).replace('\\n', '\n').replace('\\"', '"').strip()
                return {"action":"chat","params":{},"confidence":0.5,"speech":speech}
            # Attempt 4: if raw is plain prose (LLM ignored JSON instruction), use as-is
            stripped = raw.strip().strip('`')
            if not stripped.startswith('{'):
                return {"action":"chat","params":{},"confidence":0.5,"speech":stripped}
            # Last resort: show a safe fallback rather than dumping raw JSON
            return {"action":"chat","params":{},"confidence":0.5,
                    "speech":"I heard you, Boss, but had trouble processing that. Could you rephrase?"}
        except Exception as e:
            logger.error(f"LLM: {e}"); return None

    # ── TTS ───────────────────────────────────────────────────────────────────
    def _speak(self, text: str):
        self._last_speech = text
        threading.Thread(target=self._speak_async, args=(text,), daemon=True).start()

    def _speak_async(self, text: str):
        self._speaking = True
        try:
            if self._eleven:
                try:
                    audio = self._eleven.generate(text=text, voice="Rachel",
                                                  model="eleven_monolingual_v1")
                    _eleven_play(audio); return
                except Exception as e:
                    logger.warning(f"ElevenLabs: {e}")
            if self._tts:
                try:
                    self._tts.say(text); self._tts.runAndWait()
                except Exception as e:
                    logger.warning(f"pyttsx3: {e}")
        finally:
            time.sleep(0.6)   # brief cooldown so mic doesn't pick up last syllables
            self._speaking = False

    # ── Public API ─────────────────────────────────────────────────────────────
    def push_to_talk(self):
        threading.Thread(target=self._handle_voice_turn, daemon=True).start()

    def text_input(self, text: str):
        """Chat-only path: queries the chat LLM and displays the response. No TTS, no commands."""
        def _run():
            self._chat("loading", "")
            response = self._query_chat_llm(text)
            self._chat("wednesday", response)
        threading.Thread(target=_run, daemon=True).start()

    def boot_greeting(self, username: str = "Boss"):
        def _run():
            import random
            h   = datetime.now().hour
            tod = "morning" if h < 12 else "afternoon" if h < 17 else "evening"
            greetings = [
                f"Good {tod}, {username}. Clap twice to give a voice command.",
                f"Hey {username}, {tod} — all systems online. Clap twice when ready.",
                f"Welcome back, {username}. Good {tod}. Clap twice to start.",
                f"Good {tod}! {username}, I'm ready. Clap twice or type in the chat.",
            ]
            msg = random.choice(greetings)
            self._add_to_memory("System boot initiated.", msg)
            self._sphere("speaking")
            self._speak(msg)   # audio-only greeting, chat panel stays clean
            self._sphere("idle")
        threading.Thread(target=_run, daemon=True).start()

    # ── Memory ─────────────────────────────────────────────────────────────────
    def _load_memory(self):
        # We start fresh each session so the LLM doesn't reference old messages
        # that aren't visible in the UI.
        return []

    def load_persisted_memory(self) -> list:
        """Read the persisted memory file (used by Mission Control viewer)."""
        try:
            if os.path.exists(self._memory_path):
                with open(self._memory_path, "r", encoding="utf-8") as f:
                    return json.load(f) or []
        except Exception as e:
            logger.warning(f"Load persisted memory failed: {e}")
        return []

    def clear_memory(self):
        """Wipe the in-memory log and persisted JSON file."""
        self._history = []
        try:
            os.makedirs(os.path.dirname(self._memory_path), exist_ok=True)
            with open(self._memory_path, "w", encoding="utf-8") as f:
                json.dump([], f)
        except Exception as e:
            logger.warning(f"Clear memory failed: {e}")

    def _add_to_memory(self, user: str, wed: str):
        self._history.append({"user":user,"wednesday":wed,"t":time.strftime("%Y-%m-%dT%H:%M:%S")})
        if len(self._history) > self._memory_max*2:
            self._history = self._history[-self._memory_max:]
        os.makedirs(os.path.dirname(self._memory_path), exist_ok=True)
        with open(self._memory_path,"w") as f: json.dump(self._history,f,indent=2)

    # ── Compat ─────────────────────────────────────────────────────────────────
    def get_last_command(self):
        with self._lock:
            c = self._last_command; self._last_command = None; return c

    def listen_loop(self):
        self.start_wake_word_loop()
        while not self.stop_requested: time.sleep(0.5)

    def stop(self):
        self.stop_requested = True
        self._set_listening_active(False)
        try: self._cmd_queue.put_nowait(None)
        except Exception: pass
        try: self._stt_queue.put_nowait(None)
        except Exception: pass


FridayEngine = WednesdayEngine