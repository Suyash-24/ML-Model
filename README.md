# 👁️ Eyecon — AI-Powered Multimodal Interaction System

> Control your computer using **eye movements**, **hand gestures**, and **voice commands** — all unified through an AI decision layer with user authentication and adaptive tuning.

---

## ✨ Key Features

- **PyQt6 GUI Dashboard** with live camera feed, gesture overlay, status indicators
- **User Authentication** - Local SQLite auth with sign-up/login profiles
- **8 Hand Gestures** - Pinch, scroll, volume, pause/resume, and more
- **36 Voice Commands** - Navigation, mouse control, keyboard shortcuts, system control
- **Eye Tracking** - Gaze control with dwell-click and blink detection
- **AI Decision Engine** - Priority-based multi-modal arbitration (VOICE > GESTURE > EYE)
- **Real-time Calibration** - Per-user eye-tracker calibration with accuracy metrics
- **Advanced Gesture Tuning** - Separate cooldowns, edge detection, motion smoothing

---

## 📁 Project Structure

```
eyecon/
├── main.py                   ← CLI/GUI launcher (start here)
├── app.py                    ← PyQt6 main dashboard
├── auth.py                   ← User authentication & session management
├── auth_window.py            ← Login/Signup UI
├── requirements.txt          ← Python dependencies
├── config/
│   ├── settings.json         ← All tunable parameters
│   └── commands.json         ← Custom voice commands
├── modules/
│   ├── eye_tracker.py        ← Gaze estimation & blink/dwell click
│   ├── gesture_engine.py     ← Hand gesture recognition & mapping
│   ├── voice_engine.py       ← Speech-to-text & command dispatch
│   ├── ai_decision.py        ← Multi-modal priority arbitration
│   └── feedback.py           ← TTS, beep, visual flash
├── utils/
│   ├── config.py             ← Config loader with defaults
│   ├── logger.py             ← Unified logging (file + console)
│   └── auth.py               ← Auth module wrapper
└── data/
    └── eyecon.db             ← SQLite user database
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.11 (required for MediaPipe stability)
- Webcam + Microphone (microphone optional for voice)

### 1. Clone & Setup
```bash
git clone https://github.com/Suyash-24/ML-Model.git
cd eyecon
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> For voice commands, also install PyAudio:
> - **Windows**: `pip install pipwin && pipwin install pyaudio`
> - **macOS**: `brew install portaudio && pip install pyaudio`
> - **Linux**: `sudo apt install portaudio19-dev && pip install pyaudio`

### 3. Run Eyecon
```bash
# GUI mode (recommended)
python main.py --gui

# Or CLI mode
python main.py --cli
```

First run: Sign up with email → Eye calibration (9-point grid) → Ready to use!

### 4. Control your computer
- **Pause/Resume**: Show closed fist to pause, open palm to resume
- **Click**: Pinch thumb + index
- **Scroll**: Two fingers + move vertically
- **Volume**: Three fingers (down), four fingers (up)
- **Say commands**: "Open Chrome", "Take Screenshot", "Scroll down", etc.

### 5. Quit
Press `Q` in GUI or say `"Stop Eyecon"`

---

## 👀 Eye Tracking

| Feature | Description |
|---|---|
| Gaze cursor | Iris position maps to mouse cursor in real time |
| Dwell click | Stare at a point for ~0.8 s → left click |
| Blink click | Rapid blink → left click |
| Calibration | 9-point grid, polynomial mapping for accuracy |
| Smoothing | Rolling average over 6 frames reduces jitter |
| Per-user profiles | Each user has stored calibration data |

**Config keys:** `blink_click`, `dwell_click`, `gaze_smooth_frames`, `eye_moves_cursor`

---

## ✋ Hand Gestures (Optimized Detection)

| Gesture | Action | Notes |
|---|---|---|
| ✋ **OPEN_PALM** | Resume/Activate | Default at startup (system paused) |
| ✊ **FIST** | Pause/Stop | Edge detection (0.04 margin) + frame counter (6 frames) |
| 🤏 **PINCH** | Left click | 2-frame confirm, 6-frame cooldown (back-to-back reliable) |
| ✌️ **TWO_FINGERS** | Scroll | Motion accumulation buffer, smoothing factor 0.4 |
| 👉 **THREE_FINGERS** | Volume down | Immediate action |
| ☝️ **FOUR_FINGERS** | Volume up | Immediate action |
| 🎯 **POINTING** | Directional control | Point + move |
| 👈/👉 **SWIPE_LEFT/RIGHT** | Navigation | Browser tab switching |

**Latest Improvements:**
- Pinch: Separate fast cooldown (2 confirm, 6 cooldown) for back-to-back clicks
- Scroll: Accumulation buffer + smoothing (0.4 factor) for natural motion
- Fist: Edge margin detection (0.04) + hand-present frame counter (6 frames) prevents accidental triggers
- Hand re-entry: Cooldown cleared on hand exit, allows gestures immediately after re-entry

---

## 🎙️ Voice Commands (36 Total)

### Navigation
- "Scroll down / up" - Page scroll
- "Scroll left / right" - Horizontal scroll
- "Switch tab / Next tab / Previous tab" - Browser tabs
- "Go back / Go forward" - Browser navigation
- "Page down / Page up" - Page scroll

### Mouse Control
- "Click / Left click" - Mouse click
- "Right click / Double click" - Right & double click
- "Move mouse [up/down/left/right]" - Mouse movement

### Keyboard Shortcuts
- "Copy / Paste / Undo / Redo" - Clipboard & undo
- "Press [key]" - Generic key press
- "Enter / Escape / Space / Backspace" - Common keys

### System Control
- "Take screenshot" - Save to ~/Pictures
- "Increase / Decrease volume" - Volume control
- "Volume mute" - Mute/unmute
- "Open Chrome / Explorer / Settings" - Application launch

### Eyecon Control
- "Pause Eyecon / Resume Eyecon" - State control
- "Calibrate" - Re-run eye calibration
- "Help" - Show command list
- "Stop Eyecon" - Graceful shutdown

Add custom commands in `config/commands.json`.

---

## 🔐 Authentication & Profiles

- **Sign-Up**: Email, password, age, use-case, disability status, country
- **Security**: bcrypt password hashing (SHA-256 fallback)
- **Sessions**: Per-user session tracking with command analytics
- **Calibration**: Per-user eye-tracker calibration matrices saved to database
- **"Remember Me"**: Persistent login token for quick return access

---

## 🧠 AI Decision Module

The AI module acts as the brain, arbitrating between the three input channels:

```
Priority: VOICE (3) > GESTURE (2) > EYE (1)
```

**Context rules:**
- If voice command fires → gesture detection is briefly suppressed
- If eye tracking goes inactive → auto-switches to gesture-only mode
- Noise filter: same action repeated 4+ times in 5 frames → suppressed
- Cooldown gate: non-voice actions have a minimum gap between firing
- **Scroll exemption**: Scroll actions bypass noise filter for smooth motion

---

## ⚙️ Configuration (`config/settings.json`)

| Key | Default | Description |
|---|---|---|
| `gesture_confirm_frames` | 8 | Frames to confirm a gesture |
| `gesture_cooldown_frames` | 20 | Gap between gesture actions |
| `pinch_confirm_frames` | 2 | Quick pinch confirmation |
| `pinch_cooldown_frames` | 6 | Fast pinch cooldown (back-to-back clicks) |
| `pinch_distance_thresh` | 0.09 | Pinch detection distance threshold |
| `fist_confirm_frames` | 6 | Frames to confirm fist pause |
| `fist_min_present_frames` | 6 | Frames hand must be present for pause |
| `fist_edge_margin` | 0.04 | Edge detection margin (prevents accidental triggers) |
| `gesture_scroll_sensitivity` | 1200 | Scroll speed multiplier |
| `gesture_scroll_smooth` | 0.4 | Scroll smoothing factor (0-1) |
| `gesture_scroll_min_scroll` | 1 | Minimum scroll delta |
| `gesture_scroll_confirm_frames` | 2 | Frames to confirm scroll start |
| `blink_click` | true | Enable blink-to-click |
| `dwell_click` | true | Enable dwell-to-click |
| `gaze_smooth_frames` | 6 | Gaze smoothing window |
| `start_paused` | true | Start system in paused state |
| `camera_index` | 0 | Webcam device index |
| `auto_calibrate` | true | Run calibration on startup |
| `mic_energy_threshold` | 300 | Microphone sensitivity |
| `ai_action_cooldown_frames` | 15 | Min frames between AI decisions |

---

## 🔒 Safety Features

- Say **"Stop Eyecon"** or press `Q` to quit cleanly
- **Fist gesture** pauses all input immediately
- **Pause by default** - System starts paused, open palm to resume
- Accidental gesture noise filter built into AI module
- Edge margin detection prevents phantom fist triggers
- Per-gesture cooldowns prevent action spam

---

## 🐛 Known Issues & Solutions

### 1. Voice Commands Not Working
**Symptom**: Voice commands fail with "getaddrinfo failed" error
**Cause**: Requires network connectivity for Google Web Speech API
**Solutions**:
- Verify internet connection and DNS
- For offline speech recognition: `pip install vosk` and download model
- Configure offline mode in `modules/voice_engine.py`

### 2. PyAudio Missing
**Symptom**: "Could not find PyAudio; check installation"
**Cause**: PyAudio installation requires portaudio development headers
**Solutions**:
- **Windows**: `pip install pipwin && pipwin install pyaudio`
- **macOS**: `brew install portaudio && pip install pyaudio`
- **Linux**: `sudo apt install portaudio19-dev && pip install pyaudio`
- Voice commands unavailable without PyAudio (gesture/eye tracking still work)

### 3. Camera Not Opening
**Solution**: Change `camera_index` in `config/settings.json` to 1 or 2

### 4. Scroll Too Sensitive/Not Sensitive Enough
**Solution**: Adjust in `config/settings.json`:
- `gesture_scroll_sensitivity` (default 1200) - increase for slower, decrease for faster
- `gesture_scroll_smooth` (default 0.4) - increase for smoother, decrease for direct response

### 5. Pinch Not Detected / Unreliable
**Status**: Fixed with separate cooldown (2 frames confirm, 6 frames cooldown)
**If still issues**: 
- Increase `pinch_confirm_frames` (default 2)
- Decrease `pinch_distance_thresh` (default 0.09) to require closer pinch
- Increase `pinch_cooldown_frames` (default 6) if still firing too fast

### 6. Fist Pause Triggers Randomly
**Status**: Fixed with edge margin detection (0.04) and hand-present frame counter (6 frames)
**If still issues**: 
- Increase `fist_edge_margin` (default 0.04) to require hand more in center
- Increase `fist_min_present_frames` (default 6) to require longer hand presence

### 7. Gaze Accuracy Poor
**Solution**: Re-run calibration. Ensure good lighting on your face and stable head position.

### 8. Gestures Not Detected
**Solutions**:
- Ensure hand is clearly visible and well-lit
- Keep hand 30–60 cm from camera
- Check camera isn't obstructed or at odd angle
- Try adjusting `gesture_confirm_frames` in settings.json

### 9. Google OAuth Not Configured
**Status**: Patches applied but credentials/redirect URI not set up
**Impact**: Local sign-up/login works fine; OAuth is optional enhancement

---

## 📊 Performance

| Metric | Typical |
|---|---|
| GUI FPS | 25–30 |
| CPU usage | 15–30% |
| Gaze latency | <50 ms |
| Gesture latency | <40 ms |
| Voice latency | 300–800 ms (API dependent) |
| Memory usage | 200–400 MB |

---

## 🛠️ Development

### Adding Custom Gestures
Edit `modules/gesture_engine.py` → `_classify()` method to add new hand poses

### Adding Custom Voice Commands
1. Add entry to `config/commands.json`:
   ```json
   {
     "custom command": {
       "keywords": ["keyword1", "keyword2"],
       "handler": "custom_handler"
     }
   }
   ```
2. Add handler function in `modules/voice_engine.py`

### Tuning Gesture Recognition
Modify `config/settings.json` confidence thresholds and cooldown values

### Adjusting Eye Tracking Calibration
Edit `modules/eye_tracker.py` → `calibrate()` method for different calibration patterns

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Test thoroughly (especially gesture edge cases)
4. Commit with clear messages
5. Push and open Pull Request

---

## 📧 Support & Issues

For bugs, feature requests, or questions: Open a GitHub issue

---

*Built with MediaPipe · OpenCV · PyQt6 · SpeechRecognition · pyautogui · comtypes*
