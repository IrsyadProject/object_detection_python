Realtime YOLO Object Detection

Overview
- Script: `main.py`
- Purpose: realtime object detection (Ultralytics YOLO) dengan anotasi, counter objek, FPS, dan fallback deteksi tangan (OpenCV) yang memutar suara saat tangan menggenggam.

Quick setup (Windows, using the repository venv)

1) Create / activate virtual environment (PowerShell / CMD):

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

Or (Git Bash):

```bash
source .venv/Scripts/activate
```

2) Upgrade pip and install requirements

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

3) Run the script

```bash
python main.py
```

Files & configuration
- `requirements.txt`: core dependencies (`ultralytics`, `opencv-python`, `numpy`).
- `main.py` expects a WAV file called `hidup-jokowi.wav` in the project folder by default. Change `SOUND_FILE` in the script to use another file.

Notes and troubleshooting
- ModuleNotFoundError: No module named 'cv2' -> make sure you installed `opencv-python` in the same Python environment that runs `main.py`.
- If `pip install mediapipe` fails: mediapipe wheels are only available for certain Python versions/architectures. If you want MediaPipe, create a venv with Python 3.10/3.11 64-bit or use conda.

Optional improvements
- Add CLI flags `--sound-file` and `--sound-cooldown` for runtime control of the alert sound.
- Add debug mode to show the skin mask and contours for tuning the hand-fist detector.
- Use MediaPipe Hands (recommended) on a supported Python version for more robust hand detection.

If you want, I can:
- Add CLI flags (`--sound-file`, `--sound-cooldown`, `--debug-hand`).
- Add a small sample `alert.wav` (I cannot generate audio here, but I can point to sources or create instructions).
- Add a `requirements-dev.txt` or `pyproject.toml`.
