"""
modules/wednesday_engine.py  —  WEDNESDAY AI Engine (Jarvis-level)
Wake word: "wednesday" via Whisper Tiny (free, offline, no API key)
"""
import os, json, time, threading, subprocess, webbrowser, shutil, math
import numpy as np
from datetime import datetime
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
- run_command: {"cmd": "...", "shell": true} → run a shell/terminal command
- media_play_pause / media_next / media_prev / media_stop: {}
- volume: {"direction": "up|down", "steps": 3}
- shutdown: {"delay": 0} / restart: {"delay": 0} / sleep: {} / lock_screen: {}
- open_file: {"path": "..."} / open_folder: {"folder": "downloads|documents|desktop|pictures|music"}
- type_text: {"text": "...", "press_enter": false}
- tell_time / tell_date / calculate: {"expression": "..."} / tell_weather: {"city": "..."}
- eye_enable: {"enabled": true|false} / gesture_enable: {"enabled": true|false}
- clarify: {} → ONLY use this if input is truly random gibberish with zero guessable intent

DECISION EXAMPLES:
- "open youtube" / "open your tube" → {"action":"open_url","params":{"url":"https://www.youtube.com"},"confidence":0.97,"speech":"Opening YouTube, Boss."}
- "open vs code" / "open b.s. code" → {"action":"open_app","params":{"app":"vs_code"},"confidence":0.97,"speech":"Launching VS Code, Boss."}
- "open chrome" → {"action":"open_app","params":{"app":"chrome"},"confidence":0.97,"speech":"Opening Chrome, Boss."}
- "volume up" → {"action":"volume","params":{"direction":"up","steps":3},"confidence":0.99,"speech":"Turning up the volume."}
- "what time is it" → {"action":"tell_time","params":{},"confidence":0.99,"speech":"Let me check the time."}

RULES:
- Confidence threshold to execute is 0.45 — do NOT use clarify for anything above this
- For system commands: keep speech SHORT (1 sentence max, e.g. "On it, Boss.")
- For chat: give a full helpful response in speech
- NEVER leave speech empty
- The wake word is "wednesday" — user commands come after it, strip it from your understanding
- MULTI-STEP REQUESTS (e.g. "open VS Code AND create a folder"): pick the FIRST action only (e.g. open_app), mention the rest briefly in speech. Never output more than ONE JSON object.
- CODING/WRITING REQUESTS: always use the 'chat' action and write the code/text in the speech field.
"""


class WednesdayEngine:
    WAKE_WHISPER_MODEL = "base"   # Use base for both wake + STT (tiny too inaccurate)
    STT_WHISPER_MODEL  = "base"
    LLM_MODEL          = "llama3.2:3b"
    SAMPLE_RATE        = 16000
    WAKE_CHUNK_SECS    = 2.5
    CMD_MAX_SECS       = 5

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
        self._history     = self._load_memory()

        self._sphere_cb = None
        self._chat_cb   = None
        self._page_cb   = None
        self._username  = config.get("username", "Boss")
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
    def set_sphere_callback(self, fn): self._sphere_cb = fn
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
        if self._sphere_cb:
            try: self._sphere_cb(s)
            except: pass

    def _chat(self, role, text):
        if self._chat_cb:
            try: self._chat_cb(role, text)
            except: pass

    # ── Fast-path matcher ──────────────────────────────────────────────────────────
    def _fast_match(self, text: str):
        """Return a pre-built result dict if text matches a known keyword intent.
        Returns None if no fast match — caller should fall back to LLM."""
        t = text.lower().strip()
        for keywords, action, params, speech in _FAST_INTENTS:
            if any(kw in t for kw in keywords):
                return {"action": action, "params": params,
                        "confidence": 0.99, "speech": speech}
        return None

    # ── Wake word loop ─────────────────────────────────────────────────────────
    def start_wake_word_loop(self):
        """Start double-clap wake word detector (Whisper-free, TTS-immune)."""
        threading.Thread(target=self._clap_loop, daemon=True).start()
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
                            threading.Thread(target=self._handle_voice_turn,
                                             daemon=True).start()
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
                self._handle_voice_turn()

    # ── Voice pipeline ─────────────────────────────────────────────────────────────
    def _handle_voice_turn(self):
        """Record a new command after the wake word was detected alone."""
        if self._in_turn: return
        self._in_turn = True
        try:
            self._sphere("listening")
            audio = self._record_audio(self.CMD_MAX_SECS)
            if audio is None:
                self._sphere("idle"); return
            self._sphere("processing")
            transcript = self._transcribe_full(audio)
            if not transcript or len(transcript.strip()) < 3:
                self._sphere("idle"); return
            # Strip wake word if it leaked into the recording
            cleaned = transcript.strip()
            for w in _WAKE_VARIANTS:
                if cleaned.lower().startswith(w):
                    cleaned = cleaned[len(w):].strip(", "); break
            if not cleaned:
                self._sphere("idle"); return
            self._run_voice_command(cleaned)
        finally:
            self._sphere("idle")
            self._in_turn = False

    def _run_voice_command(self, text: str):
        """Process a transcribed voice command — fast path first, LLM fallback."""
        if not text or len(text.strip()) < 3: return
        was_owner = False
        if not self._in_turn:
            self._in_turn = True
            was_owner = True
        try:
            self._chat("user", text)
            # 1. Fast-path — instant, no LLM
            result = self._fast_match(text)
            if result:
                logger.info(f"Fast match → {result['action']}")
            else:
                # 2. LLM fallback
                self._sphere("processing")
                self._chat("loading", "")
                result = self._query_llm(text)
            if result is None:
                self._speak("Couldn't process that, Boss.")
                return
            self._dispatch(result, text, source="voice")
        finally:
            if was_owner:
                self._sphere("idle")
                self._in_turn = False

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
            self._sphere("speaking")
            self._chat("wednesday", speech)
            if source == "voice":
                self._speak(speech)
        if action not in ("chat", "clarify", "tell_time", "tell_date", "calculate") and confidence >= 0.45:
            self._execute(action, params)
            with self._lock: self._last_command = action
        elif action == "tell_time":
            t = datetime.now().strftime("%I:%M %p")
            msg = f"It's {t}, Boss."
            self._chat("wednesday", msg)
            if source == "voice": self._speak(msg)
        elif action == "tell_date":
            d = datetime.now().strftime("%A, %B %d %Y")
            msg = f"Today is {d}."
            self._chat("wednesday", msg)
            if source == "voice": self._speak(msg)
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

            logger.info(f"Executed: {action}")
            self._emit_history(action, params, True)
        except Exception as e:
            logger.error(f"Execute error ({action}): {e}")
            self._emit_history(action, params, False, str(e))
            self._chat("wednesday",
                       f"Sorry Boss, I couldn't complete '{action}': {e}")

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
    def _record_audio(self, max_secs=5):
        """Record audio with simple VAD: stop early after ~1s of silence."""
        if not _SD_OK: return None
        try:
            chunk_secs  = 0.4
            chunk_size  = int(self.SAMPLE_RATE * chunk_secs)
            silence_rms = 0.008   # below this = silence
            max_silence = 3       # consecutive silent chunks before stopping (~1.2s)
            max_chunks  = int(max_secs / chunk_secs)
            min_chunks  = 3       # always record at least ~1.2s

            chunks = []
            silent_streak = 0
            for _ in range(max_chunks):
                c = sd.rec(chunk_size, samplerate=self.SAMPLE_RATE, channels=1, dtype="float32")
                sd.wait()
                chunks.append(c.flatten())
                rms = float(np.sqrt(np.mean(c ** 2)))
                if rms < silence_rms:
                    silent_streak += 1
                    if silent_streak >= max_silence and len(chunks) >= min_chunks:
                        break
                else:
                    silent_streak = 0

            return np.concatenate(chunks) if chunks else None
        except Exception as e:
            logger.error(f"Record: {e}"); return None

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
                                options={"temperature":0.2,"num_predict":300})
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
        def _run():
            # 1. Fast-path (instant, no LLM)
            result = self._fast_match(text)
            if result:
                logger.info(f"Text fast match → {result['action']}")
            else:
                # 2. LLM fallback
                self._sphere("processing")
                self._chat("loading", "")
                result = self._query_llm(text)
            if result: self._dispatch(result, text, source="chat")
            self._sphere("idle")
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
                f"Good {tod}! {username}, I'm ready. Clap twice or type a command.",
            ]
            msg = random.choice(greetings)
            self._add_to_memory("System boot initiated.", msg)
            self._sphere("speaking")
            self._chat("wednesday", msg)
            self._speak(msg)
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

    def stop(self): self.stop_requested = True


FridayEngine = WednesdayEngine