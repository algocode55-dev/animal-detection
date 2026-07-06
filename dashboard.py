"""
dashboard.py — Production Animal Detection Dashboard
PyQt6 + OpenCV + YOLOv8 | Cross-platform (Linux / Windows / macOS)

Architecture:
  VideoProcessorThread  — capture + inference + filter (never blocks GUI)
  AlertThread           — audio playback queue (from utils/audio_alerts.py)
  AnimalDetectionDashboard — QMainWindow with 3-column premium dark UI

Usage:
  python dashboard.py              # normal launch (simulation mode)
  python dashboard.py --test       # diagnostics mode
  python dashboard.py --source webcam
  python dashboard.py --model path/to/best.pt
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import (
    QSize, Qt, QThread, QTimer, QUrl, pyqtSignal,
)
from PyQt6.QtGui import (
    QColor, QFont, QIcon, QImage, QKeySequence, QPixmap, QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame,
    QHBoxLayout, QHeaderView, QLabel, QMainWindow, QProgressBar,
    QPushButton, QScrollArea, QSizePolicy, QSlider, QSplitter,
    QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

# ── Local utils ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import cfg
from utils.fps_counter import FPSCounter
from utils.vision_filters import (
    apply_thermal, apply_night_vision,
    apply_contrast_boost, apply_edge_enhance,
)
from utils.drawing import draw_detection_box, draw_hud_overlay, draw_scan_line
from utils.platform_utils import open_camera, get_cuda_info, enumerate_cameras
from utils.audio_alerts import AlertThread
from utils.usb_led import led_controller

# ── Optional YOLO ─────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  VIDEO PROCESSOR THREAD
# ══════════════════════════════════════════════════════════════════════════════

class VideoProcessorThread(QThread):
    """
    Runs in a dedicated OS thread:
      • Captures frames (webcam / video file / simulation)
      • Runs YOLOv8 inference
      • Applies vision filters
      • Draws overlays
      • Emits QImage + detection metadata
    The GUI thread NEVER touches OpenCV or YOLO.
    """

    frame_ready     = pyqtSignal(QImage)
    stats_ready     = pyqtSignal(float, float, int, str)   # fps, latency_ms, count, device
    detection_event = pyqtSignal(list)                      # list[dict] per frame

    def __init__(self, parent=None):
        super().__init__(parent)

        self.running       = False
        self.source        = "webcam"   # "webcam" | "file"
        self.cam_index     = 0
        self.video_path    = ""

        # ── Tunable settings (written from GUI thread — atomic Python ints/floats) ──
        self.confidence_threshold = cfg.confidence_threshold
        self.night_vision   = False
        self.thermal_vision = False
        self.enhance_vision = False
        self.edge_enhance   = False
        self.show_scan_line = True

        # ── Internal ──────────────────────────────────────────────────────────
        self.model         = None
        self._fps          = FPSCounter(cfg.fps_window_size)
        self._sim_tick     = 0
        self._animal_x     = -100
        self._animal_y     = 280
        self._animal_speed = 4
        self._animal_type  = "Deer"
        self._animal_vis   = False
        self._total_count  = 0

        # ── Recording ─────────────────────────────────────────────────────────
        self.recording      = False
        self._writer: cv2.VideoWriter | None = None
        self.active_tracks: list[dict] = []

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_model(self) -> bool:
        if not YOLO_AVAILABLE:
            print("[WARN] ultralytics not installed — simulation mode only.")
            return False
        try:
            path = cfg.model_path
            if not os.path.exists(path):
                print(f"[WARN] {path} not found — loading {cfg.fallback_model}")
                path = cfg.fallback_model
            self.model = YOLO(path)
            print(f"[INFO] Loaded model: {path}")

            # Auto-detect device
            cuda_info = get_cuda_info()
            if cuda_info["available"] and cfg.use_half_precision:
                self.model.to("cuda")
                self.model.half()
                print(f"[INFO] Running on GPU: {cuda_info['device_name']} (FP16)")
            else:
                print("[INFO] Running on CPU")

            # Warmup
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            for _ in range(cfg.warmup_frames):
                self.model(dummy, verbose=False)
            print("[INFO] Model warmup complete.")
            return True
        except Exception as exc:
            print(f"[ERROR] Model load failed: {exc}")
            self.model = None
            return False

    # ── Thread entry point ────────────────────────────────────────────────────

    def run(self):
        self.running = True
        self._fps.reset()
        self.load_model()

        cap: cv2.VideoCapture | None = None
        if self.source == "webcam":
            cap = open_camera(self.cam_index)
            if not cap.isOpened():
                print(f"[WARN] Camera {self.cam_index} unavailable.")

        skip_next = False
        detections: list[dict] = []

        while self.running:
            loop_start = time.perf_counter()

            # ── 1. Grab frame ─────────────────────────────────────────────────
            frame: np.ndarray | None = None

            if cap is None or not cap.isOpened():
                frame = self._gen_error_frame(f"NO CAMERA SIGNAL (CAM {self.cam_index})")
                time.sleep(0.033)  # Match ~30 FPS
            else:
                ret, frame = cap.read()
                if not ret:
                    frame = self._gen_error_frame(f"SIGNAL LOSS (CAM {self.cam_index})")
                    time.sleep(0.033)

            if frame is None:
                continue

            # ── 2. Inference & Tracking ──────────────────────────────────────
            inf_latency = 0.0

            if skip_next:
                skip_next = False
            else:
                inf_start = time.perf_counter()
                new_detections = self._run_inference(frame)
                inf_latency = (time.perf_counter() - inf_start) * 1000.0

                # Frame skipping: if inference lagged, skip next decode
                if inf_latency > cfg.frame_skip_latency_ms:
                    skip_next = True

                # --- TRACKING LOGIC ---
                # 1. Decrement TTL for all active tracks
                for track in self.active_tracks:
                    track["ttl"] -= 1

                # 2. Match new detections to active tracks
                matched_track_ids = set()
                for new_det in new_detections:
                    best_track = None
                    best_iou = -1.0
                    for track in self.active_tracks:
                        if id(track) in matched_track_ids:
                            continue
                        if track["class"] == new_det["class"] and track["ttl"] > 0:
                            iou = self._calculate_iou(track["box_abs"], new_det["box_abs"])
                            if iou > best_iou:
                                best_iou = iou
                                best_track = track

                    if best_track is not None and best_iou > 0.20:
                        # Match found: update tracking coordinates and confidence, reset TTL
                        best_track["box_abs"] = new_det["box_abs"]
                        best_track["conf"] = new_det["conf"]
                        best_track["ttl"] = cfg.track_max_ttl
                        
                        # Update history
                        if "history" not in best_track:
                            best_track["history"] = [True] * cfg.track_confirm_frames
                        else:
                            best_track["history"].append(True)
                        if len(best_track["history"]) > cfg.track_history_len:
                            best_track["history"].pop(0)
                        
                        # Confirm if we have enough detections in window
                        if best_track["history"].count(True) >= cfg.track_confirm_frames:
                            best_track["confirmed"] = True
                            
                        matched_track_ids.add(id(best_track))
                    else:
                        # No match: register a new track
                        self.active_tracks.append({
                            "class": new_det["class"],
                            "conf": new_det["conf"],
                            "box_abs": new_det["box_abs"],
                            "ttl": cfg.track_max_ttl,
                            "history": [True],
                            "confirmed": False
                        })

                # 3. For tracks NOT matched in this frame, append False to history
                for track in self.active_tracks:
                    if id(track) not in matched_track_ids:
                        if "history" not in track:
                            track["history"] = [False]
                        else:
                            track["history"].append(False)
                        if len(track["history"]) > cfg.track_history_len:
                            track["history"].pop(0)
                        
                        # Drop confirmation if track has completely faded
                        if track["history"].count(True) < 1:
                            track["confirmed"] = False

                # 4. Clean up dead tracks (TTL <= 0)
                self.active_tracks = [t for t in self.active_tracks if t["ttl"] > 0]

                # 5. Expose ONLY confirmed tracks to drawing & alerting
                detections = [
                    {
                        "class": t["class"],
                        "conf": t["conf"],
                        "box_abs": t["box_abs"]
                    }
                    for t in self.active_tracks
                    if t.get("confirmed", False)
                ]

            self._total_count += len(detections)

            # ── 3. Vision filters ─────────────────────────────────────────────
            out = frame.copy()
            if self.enhance_vision:
                out = apply_contrast_boost(out)
            if self.thermal_vision:
                out = apply_thermal(out)
            elif self.night_vision:
                out = apply_night_vision(out)
            if self.edge_enhance:
                out = apply_edge_enhance(out)

            # ── 4. Draw bounding boxes ────────────────────────────────────────
            h_orig, w_orig = frame.shape[:2]
            for det in detections:
                x1, y1, x2, y2 = det["box_abs"]
                draw_detection_box(out, x1, y1, x2, y2, det["class"], det["conf"])

            # ── 5. HUD overlay ────────────────────────────────────────────────
            self._fps.tick()
            fps = self._fps.get_fps()
            mode_flags = {
                "thermal": self.thermal_vision,
                "night":   self.night_vision,
                "boost":   self.enhance_vision,
                "edge":    self.edge_enhance,
            }
            draw_hud_overlay(out, fps, len(detections), mode_flags)
            if self.show_scan_line:
                draw_scan_line(out, self._sim_tick)

            # ── 6. Recording ──────────────────────────────────────────────────
            if self.recording:
                self._write_frame(out)

            # ── 7. Emit ───────────────────────────────────────────────────────
            q_img = self._to_qimage(out)
            self.frame_ready.emit(q_img)

            device_str = self._device_str()
            self.stats_ready.emit(fps, inf_latency, self._total_count, device_str)
            self.detection_event.emit(detections)

            # ── 8. Throttle ───────────────────────────────────────────────────
            elapsed = time.perf_counter() - loop_start
            sleep_t = max(0.001, (1.0 / cfg.target_fps) - elapsed)
            time.sleep(sleep_t)

        if cap is not None:
            cap.release()
        self._stop_recording()

    # ── Inference helper ──────────────────────────────────────────────────────

    def _calculate_iou(self, box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
        """Calculate the Intersection-over-Union (IoU) of two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def _run_inference(self, frame: np.ndarray) -> list[dict]:
        detections: list[dict] = []
        h, w = frame.shape[:2]

        if self.model is not None and self.source != "simulation":
            # Optionally downscale for faster inference
            scale = cfg.inference_input_width / w if w > cfg.inference_input_width else 1.0
            inf_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale) if scale < 1.0 else frame
            results = self.model(inf_frame, verbose=False,
                                 conf=self.confidence_threshold,
                                 iou=cfg.inference_iou_threshold,
                                 agnostic_nms=True)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = [int(v / scale) for v in box.xyxy[0].tolist()]
                    conf = float(box.conf[0])
                    cls  = int(box.cls[0])
                    raw_name = self.model.names[cls].capitalize()
                    print(f"--- YOLO SAW: {raw_name} at {conf*100:.1f}% confidence ---")
                    
                    # Since best.pt is a custom dataset of wild animals, 
                    # we accept EVERYTHING except "person".
                    if "person" in raw_name.lower():
                        print(f"[DEBUG] Ignored detection '{raw_name}' (Person filtered out).")
                        continue
                        
                    detections.append({
                        "class": "Animal", "conf": conf,
                        "box_abs": (x1, y1, x2, y2),
                    })
        elif self.source == "simulation" and self._animal_vis:
            ax, ay = int(self._animal_x), int(self._animal_y)
            detections.append({
                "class": self._animal_type,
                "conf":  0.91 if self._sim_tick % 10 != 0 else 0.52,
                "box_abs": (ax, ay, ax + 100, ay + 80),
            })
        return detections

    # ── Error frame generator ─────────────────────────────────────────────────

    def _gen_error_frame(self, message: str = "NO CAMERA SIGNAL") -> np.ndarray:
        w, h = 640, 480
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Dark grid background pattern
        grid_size = 40
        for x in range(0, w, grid_size):
            cv2.line(img, (x, 0), (x, h), (12, 16, 24), 1)
        for y in range(0, h, grid_size):
            cv2.line(img, (0, y), (w, y), (12, 16, 24), 1)

        # Warning colors (reddish/amber accent)
        accent_color = (60, 60, 220)  # Soft red in BGR
        text_color = (200, 200, 200)

        # Draw techy outer border corners
        lc = 20
        cv2.line(img, (15, 15), (15 + lc, 15), accent_color, 2)
        cv2.line(img, (15, 15), (15, 15 + lc), accent_color, 2)
        cv2.line(img, (w - 15, 15), (w - 15 - lc, 15), accent_color, 2)
        cv2.line(img, (w - 15, 15), (w - 15, 15 + lc), accent_color, 2)
        cv2.line(img, (15, h - 15), (15 + lc, h - 15), accent_color, 2)
        cv2.line(img, (15, h - 15), (15, h - 15 - lc), accent_color, 2)
        cv2.line(img, (w - 15, h - 15), (w - 15 - lc, h - 15), accent_color, 2)
        cv2.line(img, (w - 15, h - 15), (w - 15, h - 15 - lc), accent_color, 2)

        # Large central warning sign
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "!", (w // 2 - 12, h // 2 - 25), font, 1.8, accent_color, 3, cv2.LINE_AA)

        # Display the main status message
        (msg_w, msg_h), _ = cv2.getTextSize(message, font, 0.65, 2)
        cv2.putText(img, message, (w // 2 - msg_w // 2, h // 2 + 15), font, 0.65, text_color, 2, cv2.LINE_AA)

        # Help text
        sub_text = "CHECK WEBCAM PHYSICAL CONNECTION"
        (sub_w, sub_h), _ = cv2.getTextSize(sub_text, font, 0.42, 1)
        cv2.putText(img, sub_text, (w // 2 - sub_w // 2, h // 2 + 45), font, 0.42, (80, 85, 95), 1, cv2.LINE_AA)

        # Tech decoration line
        cv2.line(img, (w // 2 - 100, h // 2 + 28), (w // 2 + 100, h // 2 + 28), (30, 35, 45), 1)

        return img

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_qimage(frame: np.ndarray) -> QImage:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888)

    def _device_str(self) -> str:
        if self.model is None:
            return "CPU"
        try:
            d = str(self.model.device)
            if "cuda" in d: return "GPU (CUDA)"
            if "mps"  in d: return "GPU (MPS)"
            return "CPU"
        except Exception:
            return "CPU"

    def _write_frame(self, frame: np.ndarray) -> None:
        if self._writer is None:
            h, w = frame.shape[:2]
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(cfg.recordings_dir / f"recording_{ts}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(path, fourcc, cfg.target_fps, (w, h))
            print(f"[INFO] Recording to {path}")
        self._writer.write(frame)

    def _stop_recording(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            print("[INFO] Recording saved.")

    def stop(self) -> None:
        self.running = False
        self.wait(3000)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class AnimalDetectionDashboard(QMainWindow):
    def __init__(self, start_source: str = "webcam"):
        super().__init__()
        self.setWindowTitle(cfg.window_title)
        self.setMinimumSize(cfg.min_width, cfg.min_height)

        # State
        self._is_processing   = False
        self._audio_enabled   = True
        self._recording       = False
        self._start_time      = datetime.now()
        self._log_rows: list[dict] = []

        # Threads
        self._proc = VideoProcessorThread()
        self._proc.source = start_source
        self._proc.frame_ready.connect(self._on_frame)
        self._proc.stats_ready.connect(self._on_stats)
        self._proc.detection_event.connect(self._on_detections)

        self._alert = AlertThread(cooldown_sec=cfg.alert_cooldown_sec)
        self._alert.start()

        # Session timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick_session)
        self._timer.start(1000)

        # Pulse timer for status dot
        self._pulse_state = True
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._pulse_dot)
        self._pulse_timer.start(700)

        self._load_stylesheet()
        self._build_ui()
        self._start_feed()

    # ── Stylesheet ────────────────────────────────────────────────────────────

    def _load_stylesheet(self):
        qss_path = cfg.styles_dir / "theme.qss"
        if qss_path.exists():
            self.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 10, 12, 10)
        root_layout.setSpacing(10)

        root_layout.addLayout(self._build_topbar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 0)
        root_layout.addWidget(splitter, stretch=1)

        # Keyboard shortcuts
        QShortcut(QKeySequence("F11"), self).activated.connect(self._toggle_fullscreen)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self._take_snapshot)
        QShortcut(QKeySequence("Ctrl+Q"), self).activated.connect(self.close)

    # ── Top bar ───────────────────────────────────────────────────────────────

    def _build_topbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()

        # Pulse dot
        self._dot = QLabel("●")
        self._dot.setStyleSheet("color: #10b981; font-size: 18px;")
        bar.addWidget(self._dot)

        # Title
        title = QLabel("AI ANIMAL ROADWAY DETECTION")
        title.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        title.setStyleSheet("color: #38bdf8; letter-spacing: 1px;")
        bar.addWidget(title)

        sub = QLabel("  Native Python · YOLOv8 · Real-time")
        sub.setStyleSheet("color: #475569; font-size: 11px;")
        bar.addWidget(sub)
        bar.addStretch()

        # Status badge
        self._badge = QLabel("SYSTEM READY")
        self._badge.setObjectName("badge_safe")
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bar.addWidget(self._badge)

        # Audio toggle
        self._audio_btn = QPushButton("🔊 Audio ON")
        self._audio_btn.setObjectName("btn_accent")
        self._audio_btn.clicked.connect(self._toggle_audio)
        bar.addWidget(self._audio_btn)

        # Fullscreen
        fs_btn = QPushButton("⛶ Fullscreen")
        fs_btn.clicked.connect(self._toggle_fullscreen)
        bar.addWidget(fs_btn)

        return bar

    # ── Left stats panel ──────────────────────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(240)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 6, 0)
        lay.setSpacing(10)

        # Stats card
        stats_card = self._card()
        sl = QVBoxLayout(stats_card)
        sl.setContentsMargins(14, 12, 14, 12)
        sl.setSpacing(12)
        hdr = QLabel("SYSTEM PERFORMANCE")
        hdr.setObjectName("section_title")
        sl.addWidget(hdr)

        self._lbl_fps     = self._stat_row(sl, "Frame Rate",      "— FPS",    "#10b981")
        self._lbl_lat     = self._stat_row(sl, "Inference",       "— ms",     "#38bdf8")
        self._lbl_det     = self._stat_row(sl, "Detections",      "0",        "#f43f5e")
        self._lbl_device  = self._stat_row(sl, "Device",          "—",        "#a78bfa")
        self._lbl_session = self._stat_row(sl, "Session",         "00:00",    "#fb923c")
        self._lbl_cam     = self._stat_row(sl, "Camera",          "—",        "#10b981")

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sl.addWidget(sep)

        # GPU bar
        gpu_lbl = QLabel("GPU MEMORY")
        gpu_lbl.setObjectName("section_title")
        sl.addWidget(gpu_lbl)
        self._gpu_bar = QProgressBar()
        self._gpu_bar.setRange(0, 100)
        self._gpu_bar.setValue(0)
        self._gpu_bar.setTextVisible(False)
        self._gpu_bar.setFixedHeight(8)
        sl.addWidget(self._gpu_bar)
        self._gpu_text = QLabel("N/A")
        self._gpu_text.setStyleSheet("color:#475569; font-size:10px;")
        sl.addWidget(self._gpu_text)

        lay.addWidget(stats_card)
        lay.addStretch()
        return w

    # ── Center video panel ────────────────────────────────────────────────────

    def _build_center_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # Video frame
        vid_card = self._card()
        vl = QVBoxLayout(vid_card)
        vl.setContentsMargins(6, 6, 6, 6)
        self._video_lbl = QLabel("INITIALIZING…")
        self._video_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._video_lbl.setStyleSheet("background:#060d1a; border-radius:8px; color:#334155; font-size:14px;")
        vl.addWidget(self._video_lbl)
        lay.addWidget(vid_card, stretch=1)

        # Quick action bar below video
        qbar = QHBoxLayout()
        self._btn_snap = QPushButton("📷 Snapshot")
        self._btn_snap.setObjectName("btn_snapshot")
        self._btn_snap.clicked.connect(self._take_snapshot)
        self._btn_snap.setToolTip("Ctrl+S")
        qbar.addWidget(self._btn_snap)

        self._btn_rec = QPushButton("⏺ Record")
        self._btn_rec.setObjectName("btn_record")
        self._btn_rec.clicked.connect(self._toggle_recording)
        qbar.addWidget(self._btn_rec)

        self._btn_export = QPushButton("📊 Export CSV")
        self._btn_export.setObjectName("btn_accent")
        self._btn_export.clicked.connect(self._export_csv)
        qbar.addWidget(self._btn_export)

        lay.addLayout(qbar)
        return w

    # ── Right controls + log panel ────────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(270)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 0, 0, 0)
        lay.setSpacing(10)

        tabs = QTabWidget()
        tabs.addTab(self._build_controls_tab(), "CONTROLS")
        tabs.addTab(self._build_log_tab(),      "ALERT LOG")
        lay.addWidget(tabs, stretch=1)
        return w

    def _build_controls_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(14)

        # Source selector
        src_lbl = QLabel("INPUT SOURCE")
        src_lbl.setObjectName("section_title")
        lay.addWidget(src_lbl)

        self._source_combo = QComboBox()
        self._source_combo.addItems(["Webcam (cam 0)", "Webcam (cam 1)"])
        self._source_combo.currentIndexChanged.connect(self._change_source)
        lay.addWidget(self._source_combo)

        # Start / Stop
        btn_row = QHBoxLayout()
        self._btn_start = QPushButton("▶ START")
        self._btn_start.setObjectName("btn_start")
        self._btn_start.clicked.connect(self._toggle_feed)
        btn_row.addWidget(self._btn_start)
        self._btn_stop = QPushButton("⏹ STOP")
        self._btn_stop.setObjectName("btn_stop")
        self._btn_stop.clicked.connect(self._stop_feed)
        btn_row.addWidget(self._btn_stop)
        lay.addLayout(btn_row)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        lay.addWidget(sep)

        # Confidence slider
        conf_lbl = QLabel("CONFIDENCE THRESHOLD")
        conf_lbl.setObjectName("section_title")
        lay.addWidget(conf_lbl)

        slider_row = QHBoxLayout()
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(10, 95)
        self._slider.setValue(50)
        self._slider.valueChanged.connect(self._update_conf)
        slider_row.addWidget(self._slider, stretch=1)
        self._conf_val = QLabel("50%")
        self._conf_val.setStyleSheet("color:#38bdf8; font-weight:bold; min-width:36px;")
        slider_row.addWidget(self._conf_val)
        lay.addLayout(slider_row)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        lay.addWidget(sep2)

        # Vision filters
        vis_lbl = QLabel("VISION MODES")
        vis_lbl.setObjectName("section_title")
        lay.addWidget(vis_lbl)

        self._chk_night   = QCheckBox("🌙 Night Vision Goggles")
        self._chk_thermal = QCheckBox("🌡 IR Thermal Simulation")
        self._chk_boost   = QCheckBox("⚡ Contrast Booster")
        self._chk_edge    = QCheckBox("🔬 Edge Enhancement")
        for chk in (self._chk_night, self._chk_thermal, self._chk_boost, self._chk_edge):
            lay.addWidget(chk)

        self._chk_night.stateChanged.connect(self._toggle_night)
        self._chk_thermal.stateChanged.connect(self._toggle_thermal)
        self._chk_boost.stateChanged.connect(lambda s: setattr(self._proc, "enhance_vision", bool(s)))
        self._chk_edge.stateChanged.connect(lambda s: setattr(self._proc, "edge_enhance", bool(s)))

        lay.addStretch()
        return w

    def _build_log_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)

        self._log_table = QTableWidget(0, 4)
        self._log_table.setHorizontalHeaderLabels(["Time", "Animal", "Conf", "Level"])
        self._log_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._log_table.verticalHeader().setVisible(False)
        self._log_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._log_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        lay.addWidget(self._log_table)
        return w

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _card(self) -> QFrame:
        f = QFrame()
        f.setObjectName("card")
        return f

    def _stat_row(self, layout: QVBoxLayout, label: str, value: str, color: str) -> QLabel:
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setStyleSheet("color:#475569; font-size:11px;")
        val = QLabel(value)
        val.setStyleSheet(f"color:{color}; font-weight:bold; font-size:12px;")
        val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(lbl)
        row.addWidget(val)
        layout.addLayout(row)
        return val

    # ── Signal handlers ───────────────────────────────────────────────────────

    def _on_frame(self, img: QImage):
        px = QPixmap.fromImage(img)
        px = px.scaled(self._video_lbl.size(),
                       Qt.AspectRatioMode.KeepAspectRatio,
                       Qt.TransformationMode.SmoothTransformation)
        self._video_lbl.setPixmap(px)

    def _on_stats(self, fps: float, lat: float, count: int, device: str):
        self._lbl_fps.setText(f"{fps:.1f} FPS")
        self._lbl_lat.setText(f"{lat:.1f} ms")
        self._lbl_det.setText(str(count))
        self._lbl_device.setText(device)
        self._lbl_cam.setText(f"cam {self._proc.cam_index}" if self._proc.source == "webcam" else self._proc.source)
        # GPU bar
        from utils.platform_utils import get_cuda_info
        ci = get_cuda_info()
        if ci["available"] and ci["memory_total_gb"] > 0:
            used = ci["memory_total_gb"] - ci["memory_free_gb"]
            pct  = int(100 * used / ci["memory_total_gb"])
            self._gpu_bar.setValue(pct)
            self._gpu_text.setText(f"{used:.1f}/{ci['memory_total_gb']:.1f} GB")
        else:
            self._gpu_bar.setValue(0)
            self._gpu_text.setText("No GPU")

    def _on_detections(self, detections: list):
        if not detections:
            self._badge.setText("SYSTEM SAFE")
            self._badge.setObjectName("badge_safe")
            self._badge.setStyle(self._badge.style())
            led_controller.turn_off()
            return

        self._badge.setText(f"⚠ ANIMAL DETECTED ({len(detections)})")
        self._badge.setObjectName("badge_alert")
        self._badge.setStyle(self._badge.style())

        led_controller.turn_on()

        high = any(d["conf"] >= cfg.alert_high_conf_threshold for d in detections)
        if self._audio_enabled:
            self._alert.trigger(AlertThread.PRIORITY_HIGH if high else AlertThread.PRIORITY_LOW)

        for d in detections:
            self._add_log_row(d["class"], d["conf"])

    def _add_log_row(self, name: str, conf: float):
        ts = datetime.now().strftime("%H:%M:%S")
        # Dedup: same class within cooldown window
        if self._log_table.rowCount() > 0:
            last_cls  = self._log_table.item(0, 1)
            last_time = self._log_table.item(0, 0)
            if last_cls and last_time and last_cls.text() == name:
                try:
                    dt = datetime.strptime(ts, "%H:%M:%S") - datetime.strptime(last_time.text(), "%H:%M:%S")
                    if abs(dt.total_seconds()) < cfg.log_dedup_sec:
                        self._log_table.setItem(0, 2, QTableWidgetItem(f"{conf:.0%}"))
                        return
                except Exception:
                    pass

        level = "HIGH" if conf >= cfg.alert_high_conf_threshold else "MED"
        color = QColor("#f43f5e") if conf >= cfg.alert_high_conf_threshold else QColor("#fb923c")

        self._log_table.insertRow(0)
        for col, text in enumerate([ts, name, f"{conf:.0%}", level]):
            item = QTableWidgetItem(text)
            item.setForeground(color)
            self._log_table.setItem(0, col, item)

        self._log_rows.insert(0, {"time": ts, "animal": name, "conf": conf, "level": level})

        if self._log_table.rowCount() > cfg.log_max_rows:
            self._log_table.removeRow(cfg.log_max_rows)
            self._log_rows = self._log_rows[:cfg.log_max_rows]

    # ── Controls ──────────────────────────────────────────────────────────────

    def _start_feed(self):
        self._proc.start()
        self._is_processing = True
        self._btn_start.setText("▶ RUNNING")

    def _toggle_feed(self):
        if self._is_processing:
            self._stop_feed()
        else:
            self._proc.start()
            self._is_processing = True
            self._btn_start.setText("▶ RUNNING")

    def _stop_feed(self):
        self._proc.stop()
        self._is_processing = False
        self._btn_start.setText("▶ START")
        self._badge.setText("STANDBY")
        self._badge.setObjectName("badge_safe")
        self._badge.setStyle(self._badge.style())

    def _change_source(self, idx: int):
        was_running = self._is_processing
        if was_running:
            self._proc.stop()

        if idx == 0:
            self._proc.source = "webcam"
            self._proc.cam_index = 0
        elif idx == 1:
            self._proc.source = "webcam"
            self._proc.cam_index = 1

        if was_running:
            self._proc.start()
            self._is_processing = True

    def _update_conf(self, val: int):
        self._proc.confidence_threshold = val / 100.0
        self._conf_val.setText(f"{val}%")

    def _toggle_night(self, state: int):
        self._proc.night_vision = bool(state)
        if state:
            self._chk_thermal.setChecked(False)

    def _toggle_thermal(self, state: int):
        self._proc.thermal_vision = bool(state)
        if state:
            self._chk_night.setChecked(False)

    def _toggle_audio(self):
        self._audio_enabled = not self._audio_enabled
        if self._audio_enabled:
            self._audio_btn.setText("🔊 Audio ON")
            self._audio_btn.setObjectName("btn_accent")
        else:
            self._audio_btn.setText("🔇 Audio OFF")
            self._audio_btn.setObjectName("btn_stop")
        self._audio_btn.setStyle(self._audio_btn.style())

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _toggle_recording(self):
        self._recording = not self._recording
        self._proc.recording = self._recording
        if self._recording:
            self._btn_rec.setText("⏹ Stop Rec")
            self._btn_rec.setProperty("active", "true")
        else:
            self._btn_rec.setText("⏺ Record")
            self._btn_rec.setProperty("active", "false")
        self._btn_rec.setStyle(self._btn_rec.style())

    def _take_snapshot(self):
        px = self._video_lbl.pixmap()
        if px and not px.isNull():
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(cfg.snapshots_dir / f"snapshot_{ts}.png")
            px.save(path, "PNG")
            print(f"[INFO] Snapshot saved: {path}")

    def _export_csv(self):
        if not self._log_rows:
            return
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = str(cfg.logs_dir / f"detections_{ts}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["time", "animal", "conf", "level"])
            writer.writeheader()
            writer.writerows(self._log_rows)
        print(f"[INFO] CSV exported: {path}")

    # ── Timers ────────────────────────────────────────────────────────────────

    def _tick_session(self):
        delta = datetime.now() - self._start_time
        mins, secs = divmod(int(delta.total_seconds()), 60)
        self._lbl_session.setText(f"{mins:02d}:{secs:02d}")

    def _pulse_dot(self):
        self._pulse_state = not self._pulse_state
        color = "#10b981" if (self._pulse_state and self._is_processing) else "#1e293b"
        self._dot.setStyleSheet(f"color:{color}; font-size:18px;")

    def closeEvent(self, event):
        self._proc.stop()
        self._alert.stop()
        led_controller.turn_off()
        event.accept()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_test_mode():
    """Print full diagnostics and exit."""
    import platform
    print("=" * 55)
    print("  Animal Detection Dashboard — Diagnostics")
    print("=" * 55)
    print(f"  OS          : {platform.system()} {platform.release()}")
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  OpenCV      : {cv2.__version__}")
    try:
        import numpy as np; print(f"  NumPy       : {np.__version__}")
    except ImportError:
        print("  NumPy       : MISSING")
    print(f"  YOLO        : {'OK' if YOLO_AVAILABLE else 'NOT INSTALLED'}")
    if YOLO_AVAILABLE:
        try:
            YOLO(cfg.model_path); print(f"  Model       : {cfg.model_path} ✓")
        except Exception as e:
            print(f"  Model       : {e}")
    from utils.platform_utils import get_cuda_info, enumerate_cameras
    ci = get_cuda_info()
    print(f"  CUDA        : {'✓ ' + ci['device_name'] if ci['available'] else 'Not available'}")
    cams = enumerate_cameras()
    print(f"  Cameras     : {cams if cams else 'None detected'}")
    print(f"  Audio bknd  : {AlertThread._detect_backend()}")
    app = QApplication(sys.argv)
    print("  PyQt6 GUI   : ✓ Initialized")
    print("=" * 55)
    print("  All checks complete.")
    print("=" * 55)
    import os as _os; _os._exit(0)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animal Detection Dashboard")
    parser.add_argument("--test",   action="store_true", help="Run diagnostics and exit")
    parser.add_argument("--source", default="webcam",
                        choices=["webcam"],
                        help="Initial video source")
    parser.add_argument("--model",  default=None, help="Path to YOLO .pt model")
    args = parser.parse_args()

    if args.model:
        cfg.model_path = args.model

    if args.test:
        app = QApplication(sys.argv)
        run_test_mode()

    app = QApplication(sys.argv)
    app.setApplicationName("Animal Detection Dashboard")
    window = AnimalDetectionDashboard(start_source=args.source)
    window.show()
    sys.exit(app.exec())
