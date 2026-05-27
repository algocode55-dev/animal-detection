# AI Animal Roadway Detection Dashboard

A production-quality, ultra-low-latency **Animal Detection Dashboard** built with Python, PyQt6, OpenCV and YOLOv8. Replaces a browser-based dashboard with a native desktop application achieving 30+ FPS real-time inference.

![Dashboard Preview](assets/icons/preview.png)

---

## Features

| Feature | Detail |
|---------|--------|
| **Real-time YOLO inference** | YOLOv8 via Ultralytics on CPU or CUDA GPU |
| **Vision modes** | Night Vision, IR Thermal, Contrast Boost, Edge Enhance |
| **Multi-source input** | Simulation, Webcam (0/1), Video File |
| **Alert system** | Audio alerts with cooldown — cross-platform |
| **Snapshot** | `Ctrl+S` or button — saves PNG to `logs/snapshots/` |
| **Recording** | Toggle MP4 recording to `logs/recordings/` |
| **CSV export** | One-click alert log export |
| **Fullscreen** | `F11` toggle |
| **GPU monitor** | VRAM progress bar in sidebar |
| **Session timer** | Elapsed runtime display |

---

## Project Structure

```
animal-detection-dashboard/
├── dashboard.py              # Main application entry point
├── requirements.txt
├── animal_dashboard.spec     # PyInstaller build config
├── assets/
│   ├── icons/
│   ├── sounds/
│   └── styles/
│       └── theme.qss         # QSS dark theme
├── models/
│   └── best.pt               # Custom trained YOLOv8 model
├── logs/                     # Auto-created: snapshots/, recordings/
├── backend/
│   └── best.pt               # Original model location (also searched)
└── utils/
    ├── config.py             # Central configuration
    ├── fps_counter.py        # Sliding-window FPS
    ├── vision_filters.py     # OpenCV filters
    ├── drawing.py            # Bounding box / HUD helpers
    ├── audio_alerts.py       # Cross-platform AlertThread
    └── platform_utils.py     # OS-specific camera + CUDA helpers
```

---

## Installation

### Prerequisites
- Python 3.9+
- pip

### Linux / macOS

```bash
cd animal-detection-dashboard

# Create & activate virtualenv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) GPU support — install PyTorch with CUDA
# Visit https://pytorch.org/get-started/locally/ for the correct command
```

### Windows

```powershell
cd animal-detection-dashboard
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

```bash
# Normal launch (starts in simulation mode)
python dashboard.py

# Start directly with webcam
python dashboard.py --source webcam

# Use a specific model
python dashboard.py --model path/to/custom.pt

# Run diagnostics / test mode
python dashboard.py --test
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `F11` | Toggle fullscreen |
| `Ctrl+S` | Save snapshot |
| `Ctrl+Q` | Quit |

---

## Model Setup

Place your trained model at either:
- `models/best.pt`  ← preferred
- `backend/best.pt` ← also searched automatically

If neither is found, `yolov8n.pt` is downloaded from the Ultralytics hub.

---

## GPU Acceleration

The application auto-detects CUDA via PyTorch:
- If CUDA is available: model runs on GPU with FP16 (half precision)
- If not: model runs on CPU

To verify:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Building a Standalone Executable

### Linux

```bash
pip install pyinstaller
pyinstaller animal_dashboard.spec

# Output: dist/AnimalDetectionDashboard
chmod +x dist/AnimalDetectionDashboard
./dist/AnimalDetectionDashboard
```

### Windows

```powershell
pip install pyinstaller
pyinstaller animal_dashboard.spec
# Output: dist\AnimalDetectionDashboard.exe
```

### One-liner (alternative)

```bash
pyinstaller --onefile --windowed \
  --add-data "assets:assets" \
  --add-data "models:models" \
  --add-data "utils:utils" \
  dashboard.py
```

---

## Cross-Platform Notes

| Concern | Solution |
|---------|----------|
| Camera backend | `CAP_DSHOW` (Windows), `CAP_V4L2` (Linux), `CAP_AVFOUNDATION` (macOS) |
| Audio | `simpleaudio` → `pygame` → `Qt QSoundEffect` → ASCII bell fallback |
| File paths | `pathlib.Path` throughout — no hardcoded separators |
| Threading | `QThread` + Python GIL-safe signals — no platform threads |
| Font | `Segoe UI` (Win) / `Ubuntu` (Linux) auto-selected |

---

## Architecture

```
┌─ GUI Thread (QMainWindow) ──────────────────────────┐
│  AnimalDetectionDashboard                           │
│  • Renders QImage via QLabel.setPixmap()            │
│  • Updates stats labels, log table, GPU bar         │
│  • Never calls OpenCV or YOLO directly              │
└──────────────┬──────────────────────────────────────┘
               │ Qt Signals (thread-safe)
   ┌───────────┴──────────┐      ┌──────────────────┐
   │ VideoProcessorThread │      │   AlertThread    │
   │  • cv2.VideoCapture  │      │  • Audio queue   │
   │  • YOLO inference    │      │  • simpleaudio   │
   │  • Vision filters    │      │  • Cooldown mgr  │
   │  • HUD drawing       │      └──────────────────┘
   │  • MP4 recording     │
   └──────────────────────┘
```

---

## License

MIT License — see LICENSE file.
