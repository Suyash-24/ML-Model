"""
setup.py  —  Eyecon FRIDAY One-Command Setup
─────────────────────────────────────────────
Run:  python setup.py

Does:
  1. Checks Python version (needs 3.11+)
  2. pip install -r requirements.txt
  3. Checks Ollama is installed and running
  4. Pulls llama3.2:3b if not present
  5. Creates data/ directory
  6. Checks webcam availability
  7. Prints setup summary
"""

import sys
import os
import subprocess
import shutil


def _run(cmd, desc, required=True):
    print(f"\n{'─'*50}")
    print(f"  {desc}")
    print(f"{'─'*50}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        if required:
            print(f"\n  ✗  FAILED: {desc}")
            print(f"     Command: {cmd}")
            sys.exit(1)
        else:
            print(f"\n  ⚠  Optional step failed: {desc}")
    else:
        print(f"\n  ✓  Done: {desc}")


def main():
    print("\n" + "═"*52)
    print("  EYECON FRIDAY  —  Setup & Installation")
    print("═"*52)

    # ── 1. Python version ─────────────────────────────────────────────
    print(f"\n  Python {sys.version}")
    if sys.version_info < (3, 11):
        print("  ✗  Python 3.11+ required")
        sys.exit(1)
    print("  ✓  Python version OK")

    # ── 2. pip install ────────────────────────────────────────────────
    _run(f"{sys.executable} -m pip install --upgrade pip",
         "Upgrading pip")
    _run(f"{sys.executable} -m pip install -r requirements.txt",
         "Installing Python dependencies")

    # ── 3. Ollama check ───────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  Checking Ollama…")
    if not shutil.which("ollama"):
        print("""
  ⚠  Ollama not found. Install it first:
     Windows/Mac: https://ollama.com/download
     Linux:  curl -fsSL https://ollama.com/install.sh | sh

  Then re-run:  python setup.py
""")
    else:
        print("  ✓  Ollama found")
        # Check if model already exists
        result = subprocess.run("ollama list", shell=True,
                                capture_output=True, text=True)
        if "llama3.2:3b" in result.stdout:
            print("  ✓  llama3.2:3b already pulled")
        else:
            print("  Pulling llama3.2:3b (≈2.0 GB)…")
            _run("ollama pull llama3.2:3b",
                 "Pulling Llama 3.2 3B model", required=False)

    # ── 4. Create directories ─────────────────────────────────────────
    print(f"\n{'─'*50}")
    for d in ["data", "logs", "config"]:
        os.makedirs(d, exist_ok=True)
        print(f"  ✓  {d}/ ready")

    # ── 5. Webcam check ───────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  Checking webcam…")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("  ✓  Webcam found (index 0)")
            cap.release()
        else:
            print("  ⚠  No webcam at index 0")
            print("     Update camera_index in config/settings.json")
    except ImportError:
        print("  ⚠  opencv-python not installed properly")

    # ── 6. PyQt6-WebEngine check ──────────────────────────────────────
    print(f"\n{'─'*50}")
    try:
        import PyQt6.QtWebEngineWidgets
        print("  ✓  PyQt6-WebEngine available (3D sphere works)")
    except ImportError:
        print("  ⚠  PyQt6-WebEngine missing")
        print("     Run: pip install PyQt6-WebEngine")

    # ── 7. API keys reminder ──────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  API keys (edit config/settings.json):")
    print("  • elevenlabs_api_key — free at elevenlabs.io")
    print("    (FRIDAY voice clone — optional, falls back to pyttsx3)")
    print("  • porcupine_key — free at picovoice.io")
    print("    (wake word 'Hey FRIDAY' — optional, use mic button instead)")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'═'*52}")
    print("  SETUP COMPLETE")
    print(f"{'═'*52}")
    print("\n  To run Eyecon FRIDAY:")
    print("    python main.py\n")
    print("  First launch tips:")
    print("  1. Complete biometric enrollment (face + hand + voice)")
    print("  2. Show OPEN PALM to activate the system")
    print("  3. Say 'Hey FRIDAY' or click mic button")
    print("  4. Eye calibration: look at each dot when it appears\n")


if __name__ == "__main__":
    main()