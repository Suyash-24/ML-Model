"""
modules/friday_engine.py
─────────────────────────
FRIDAY — AI Voice Engine
Replaces voice_engine.py entirely.

Pipeline:
  pvporcupine  →  "Hey FRIDAY" wake word (2% CPU, always-on)
  Whisper      →  Speech-to-text (GPU, offline, accurate)
  Ollama       →  Llama3.2:3b intent understanding + function calling
  ElevenLabs   →  FRIDAY voice clone TTS
  pyautogui    →  System action execution

PLACE AT: ML-Model/modules/friday_engine.py
"""

import os, sys, json, time, threading, tempfile, subprocess, queue
import numpy as np
from utils.logger import EyeconLogger

logger = EyeconLogger("FRIDAY")

# ── Optional imports (graceful degradation) ───────────────────────────────────
try:
    import whisper
    _WHISPER_OK = True
except ImportError:
    _WHISPER_OK = False
    logger.warning("openai-whisper not installed — STT disabled")

try:
    import ollama as _ollama
    _OLLAMA_OK = True
except ImportError:
    _OLLAMA_OK = False
    logger.warning("ollama not installed — LLM disabled")

try:
    import pvporcupine
    import sounddevice as sd
    _WAKE_OK = True
except ImportError:
    _WAKE_OK = False
    logger.warning("pvporcupine/sounddevice not installed — wake word disabled")

try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs import play
    _ELEVEN_OK = True
except ImportError:
    _ELEVEN_OK = False
    logger.warning("elevenlabs not installed — using pyttsx3 fallback")

try:
    import pyttsx3
    _PYTTSX3_OK = True
except ImportError:
    _PYTTSX3_OK = False

import pyautogui
pyautogui.FAILSAFE = False


# ── FRIDAY SYSTEM PROMPT ──────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are FRIDAY, the AI assistant for the Eyecon multimodal control system.
You control the user's computer via eye tracking, hand gestures, and your voice commands.
Address the user as "Boss". Be concise, professional, and slightly witty.
Never say more than 2 sentences in your speech response.

You must ALWAYS respond with a single valid JSON object — nothing else. No markdown, no explanation outside the JSON.

JSON format:
{
  "action": "<action_name>",
  "params": {},
  "confidence": 0.95,
  "speech": "<what you say out loud>"
}

Available actions and their params:
- scroll: {"direction": "up|down", "amount": 3}
- click: {"button": "left|right|double"}
- volume: {"direction": "up|down", "steps": 3}
- brightness: {"direction": "up|down", "steps": 3}
- open_app: {"app": "chrome|notepad|terminal|files"}
- switch_tab: {"direction": "next|prev"}
- screenshot: {}
- hotkey: {"keys": ["ctrl","c"]}
- type_text: {"text": "..."}
- eye_enable: {"enabled": true|false}
- gesture_enable: {"enabled": true|false}
- calibrate_eye: {}
- pause_system: {}
- resume_system: {}
- clarify: {}
- none: {}

If the user's intent is unclear or confidence < 0.70, use action "clarify" and ask them to repeat.
If no computer action is needed (just a question), use action "none" and answer in speech.
"""


class FridayEngine:
    """
    Drop-in replacement for VoiceEngine.
    Same interface expected by ai_decision.py:
      - get_last_command() → str | None
      - stop_requested (bool)
      - paused (bool)
      - stop()
    Plus new methods:
      - start_wake_word_loop()
      - set_sphere_callback(fn)   → called with state string
      - set_chat_callback(fn)     → called with (role, text)
    """

    MODEL = "llama3.2:3b"
    WHISPER_MODEL = "base"
    ELEVEN_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Replace with FRIDAY voice ID from ElevenLabs

    def __init__(self, config, feedback):
        self.cfg      = config
        self.feedback = feedback

        # State
        self.stop_requested  = False
        self.paused          = False
        self._last_command   = None
        self._last_speech    = None
        self._lock           = threading.Lock()

        # Memory — last N conversation turns
        self._memory_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "friday_memory.json")
        self._memory_max  = config.get("friday_memory_turns", 10)
        self._history     = self._load_memory()

        # Callbacks for UI
        self._sphere_cb: callable = None   # sphere state: "idle"|"listening"|"processing"|"speaking"
        self._chat_cb:   callable = None   # (role, text) → add to chat panel
        self._username   = config.get("username", "Boss")

        # Whisper model (loaded once, lazy)
        self._whisper_model = None

        # ElevenLabs client
        self._eleven = None
        if _ELEVEN_OK:
            api_key = config.get("elevenlabs_api_key", "")
            if api_key:
                self._eleven = ElevenLabs(api_key=api_key)

        # pyttsx3 fallback
        self._tts = None
        if _PYTTSX3_OK and not (self._eleven):
            try:
                self._tts = pyttsx3.init()
                self._tts.setProperty("rate", 175)
                self._tts.setProperty("volume", 0.9)
            except Exception:
                pass

        # Audio queue for wake word pipeline
        self._audio_q: queue.Queue = queue.Queue()

        logger.info("FridayEngine initialised")

    # ─────────────────────────────────────────────────────────────────────────
    #  CALLBACKS
    # ─────────────────────────────────────────────────────────────────────────
    def set_sphere_callback(self, fn):  self._sphere_cb = fn
    def set_chat_callback(self, fn):    self._chat_cb   = fn

    def _sphere(self, state: str):
        if self._sphere_cb:
            try: self._sphere_cb(state)
            except Exception: pass

    def _chat(self, role: str, text: str):
        if self._chat_cb:
            try: self._chat_cb(role, text)
            except Exception: pass

    # ─────────────────────────────────────────────────────────────────────────
    #  WAKE WORD LOOP  (background thread)
    # ─────────────────────────────────────────────────────────────────────────
    def start_wake_word_loop(self):
        t = threading.Thread(target=self._wake_loop, daemon=True)
        t.start()
        logger.info("Wake word loop started")

    def _wake_loop(self):
        if not _WAKE_OK:
            logger.warning("Wake word unavailable — falling back to push-to-talk only")
            return

        try:
            porcupine = pvporcupine.create(
                access_key=self.cfg.get("porcupine_key", ""),
                keywords=["hey friday"]
            )
        except Exception as e:
            logger.error(f"Porcupine init failed: {e}")
            return

        frame_len = porcupine.frame_length
        sample_rate = porcupine.sample_rate

        with sd.InputStream(samplerate=sample_rate, channels=1,
                            dtype='int16', blocksize=frame_len) as stream:
            while not self.stop_requested:
                if self.paused:
                    time.sleep(0.1)
                    continue
                pcm, _ = stream.read(frame_len)
                idx = porcupine.process(pcm.flatten())
                if idx >= 0:
                    logger.info("Wake word detected: Hey FRIDAY")
                    self._handle_voice_turn()

        porcupine.delete()

    # ─────────────────────────────────────────────────────────────────────────
    #  LISTEN → STT → LLM → ACTION → TTS
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_voice_turn(self):
        """Full pipeline for one voice interaction."""
        self._sphere("listening")
        self._chat("system", "Listening…")

        # 1. Record audio
        audio = self._record_audio(max_secs=6)
        if audio is None:
            self._sphere("idle")
            return

        # 2. Whisper STT
        self._sphere("processing")
        transcript = self._transcribe(audio)
        if not transcript or len(transcript.strip()) < 2:
            self._sphere("idle")
            return

        logger.info(f"Heard: '{transcript}'")
        self._chat("user", transcript)

        # 3. Ollama LLM
        result = self._query_llm(transcript)
        if result is None:
            self._speak("Sorry Boss, I couldn't process that.")
            self._sphere("idle")
            return

        action     = result.get("action",     "none")
        params     = result.get("params",     {})
        speech     = result.get("speech",     "")
        confidence = result.get("confidence", 0.0)

        logger.info(f"LLM → action={action} confidence={confidence:.2f}")

        # 4. Speak response
        if speech:
            self._sphere("speaking")
            self._chat("friday", speech)
            self._speak(speech)

        # 5. Execute action
        if action not in ("none", "clarify") and confidence >= 0.65:
            self._execute(action, params)
            with self._lock:
                self._last_command = action

        # 6. Save to memory
        self._add_to_memory(transcript, speech)
        self._sphere("idle")

    def _record_audio(self, max_secs=6, sample_rate=16000):
        """Record audio until silence or max_secs."""
        if not _WAKE_OK:
            return None
        try:
            frames = int(sample_rate * max_secs)
            audio  = sd.rec(frames, samplerate=sample_rate,
                            channels=1, dtype="float32")
            sd.wait()
            return audio.flatten()
        except Exception as e:
            logger.error(f"Audio record error: {e}")
            return None

    def _transcribe(self, audio: np.ndarray) -> str:
        """Run Whisper STT on audio array."""
        if not _WHISPER_OK:
            return ""
        if self._whisper_model is None:
            logger.info("Loading Whisper model…")
            self._whisper_model = whisper.load_model(
                self.WHISPER_MODEL, device="cuda")
        try:
            result = self._whisper_model.transcribe(
                audio, fp16=True, language="en")
            return result.get("text", "").strip()
        except Exception as e:
            logger.error(f"Whisper error: {e}")
            return ""

    def _query_llm(self, user_text: str) -> dict | None:
        """Send text to Ollama Llama3.2 and parse JSON response."""
        if not _OLLAMA_OK:
            return {"action": "none", "params": {}, "confidence": 1.0,
                    "speech": "LLM not available. Install ollama."}

        # Build messages with memory context
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for turn in self._history[-self._memory_max:]:
            messages.append({"role": "user",      "content": turn["user"]})
            messages.append({"role": "assistant",  "content": turn["friday"]})
        messages.append({"role": "user", "content": user_text})

        try:
            resp = _ollama.chat(
                model=self.MODEL,
                messages=messages,
                options={"temperature": 0.3, "num_predict": 200}
            )
            raw = resp["message"]["content"].strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            return json.loads(raw)

        except json.JSONDecodeError:
            logger.warning(f"LLM non-JSON response: {raw[:100]}")
            # Try to extract speech at least
            return {"action": "none", "params": {}, "confidence": 0.5,
                    "speech": raw[:200]}
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    #  EXECUTE ACTIONS
    # ─────────────────────────────────────────────────────────────────────────
    def _execute(self, action: str, params: dict):
        try:
            if action == "scroll":
                direction = params.get("direction", "down")
                amount    = int(params.get("amount", 3))
                pyautogui.scroll(amount if direction == "up" else -amount)

            elif action == "click":
                btn = params.get("button", "left")
                if btn == "double": pyautogui.doubleClick()
                elif btn == "right": pyautogui.rightClick()
                else: pyautogui.click()

            elif action == "volume":
                key   = "volumeup" if params.get("direction") == "up" else "volumedown"
                steps = int(params.get("steps", 3))
                for _ in range(steps): pyautogui.press(key)

            elif action == "brightness":
                key   = "brightnessup" if params.get("direction") == "up" else "brightnessdown"
                steps = int(params.get("steps", 3))
                for _ in range(steps): pyautogui.press(key)

            elif action == "open_app":
                app = params.get("app", "").lower()
                cmds = {
                    "chrome": ["google-chrome", "chromium-browser", "chrome"],
                    "notepad": ["notepad", "gedit", "nano"],
                    "terminal": ["gnome-terminal", "cmd", "powershell"],
                    "files": ["nautilus", "explorer"],
                }
                for cmd in cmds.get(app, [app]):
                    try: subprocess.Popen([cmd]); break
                    except FileNotFoundError: continue

            elif action == "switch_tab":
                if params.get("direction") == "next":
                    pyautogui.hotkey("ctrl", "tab")
                else:
                    pyautogui.hotkey("ctrl", "shift", "tab")

            elif action == "screenshot":
                ts   = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.expanduser(f"~/Pictures/friday_{ts}.png")
                pyautogui.screenshot(path)

            elif action == "hotkey":
                keys = params.get("keys", [])
                if keys: pyautogui.hotkey(*keys)

            elif action == "type_text":
                pyautogui.typewrite(params.get("text", ""), interval=0.05)

            elif action == "pause_system":
                self.paused = True

            elif action == "resume_system":
                self.paused = False

            elif action == "calibrate_eye":
                pass   # signal handled by ai_decision

            logger.info(f"Executed: {action} {params}")

        except Exception as e:
            logger.error(f"Execute error ({action}): {e}")

    # ─────────────────────────────────────────────────────────────────────────
    #  TTS
    # ─────────────────────────────────────────────────────────────────────────
    def _speak(self, text: str):
        self._last_speech = text
        threading.Thread(target=self._speak_async, args=(text,), daemon=True).start()

    def _speak_async(self, text: str):
        # Try ElevenLabs first
        if self._eleven:
            try:
                audio = self._eleven.generate(
                    text=text,
                    voice=self.ELEVEN_VOICE_ID,
                    model="eleven_monolingual_v1"
                )
                play(audio)
                return
            except Exception as e:
                logger.warning(f"ElevenLabs error: {e}")

        # pyttsx3 fallback
        if self._tts:
            try:
                self._tts.say(text)
                self._tts.runAndWait()
            except Exception as e:
                logger.warning(f"pyttsx3 error: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    #  PUSH-TO-TALK (called from UI mic button)
    # ─────────────────────────────────────────────────────────────────────────
    def push_to_talk(self):
        threading.Thread(target=self._handle_voice_turn, daemon=True).start()

    # ─────────────────────────────────────────────────────────────────────────
    #  TEXT INPUT (called from UI chat input)
    # ─────────────────────────────────────────────────────────────────────────
    def text_input(self, text: str):
        def _run():
            self._sphere("processing")
            self._chat("user", text)
            result = self._query_llm(text)
            if result:
                speech     = result.get("speech", "")
                action     = result.get("action", "none")
                params     = result.get("params", {})
                confidence = result.get("confidence", 0.0)
                if speech:
                    self._sphere("speaking")
                    self._chat("friday", speech)
                    self._speak(speech)
                if action not in ("none", "clarify") and confidence >= 0.65:
                    self._execute(action, params)
                    with self._lock:
                        self._last_command = action
                self._add_to_memory(text, speech)
            self._sphere("idle")
        threading.Thread(target=_run, daemon=True).start()

    # ─────────────────────────────────────────────────────────────────────────
    #  BOOT GREETING
    # ─────────────────────────────────────────────────────────────────────────
    def boot_greeting(self, username: str = "Boss"):
        from datetime import datetime
        h = datetime.now().hour
        tod = "morning" if h < 12 else "afternoon" if h < 17 else "evening"
        msg = f"Good {tod}, {username}. All systems are online. Standing by."
        self._chat("friday", msg)
        self._speak(msg)

    # ─────────────────────────────────────────────────────────────────────────
    #  MEMORY
    # ─────────────────────────────────────────────────────────────────────────
    def _load_memory(self) -> list:
        if os.path.exists(self._memory_path):
            try:
                with open(self._memory_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _add_to_memory(self, user_text: str, friday_text: str):
        self._history.append({"user": user_text, "friday": friday_text,
                               "t": time.strftime("%Y-%m-%dT%H:%M:%S")})
        if len(self._history) > self._memory_max * 2:
            self._history = self._history[-self._memory_max:]
        os.makedirs(os.path.dirname(self._memory_path), exist_ok=True)
        with open(self._memory_path, "w") as f:
            json.dump(self._history, f, indent=2)

    # ─────────────────────────────────────────────────────────────────────────
    #  INTERFACE COMPAT (ai_decision.py calls these)
    # ─────────────────────────────────────────────────────────────────────────
    def get_last_command(self) -> str | None:
        with self._lock:
            cmd = self._last_command
            self._last_command = None
        return cmd

    def listen_loop(self):
        """Legacy compat — start wake word loop instead."""
        self.start_wake_word_loop()
        while not self.stop_requested:
            time.sleep(0.5)

    def stop(self):
        self.stop_requested = True