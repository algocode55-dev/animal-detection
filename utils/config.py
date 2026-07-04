"""
config.py — Central configuration for Animal Detection Dashboard.
All tunable constants and path resolution live here.
"""

from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


def _resolve_model_path() -> str:
    """Search common locations for best.pt and return first found path."""
    candidates = [
        "models/best.pt",
        "backend/best.pt",
        "best.pt",
    ]
    base = Path(__file__).parent.parent  # project root
    for c in candidates:
        p = base / c
        if p.exists():
            return str(p)
    return str(base / "models" / "best.pt")  # default (may not exist yet)


@dataclass
class Config:
    # ── Model ─────────────────────────────────────────────────────────────────
    model_path: str = field(default_factory=_resolve_model_path)
    fallback_model: str = "yolov8n.pt"          # Downloaded from ultralytics hub if best.pt missing

    # ── Inference ─────────────────────────────────────────────────────────────
    confidence_threshold: float = 0.50          # Default detection confidence
    inference_input_width: int = 640            # Resize frames to this width before inference
    use_half_precision: bool = True             # FP16 when CUDA available
    warmup_frames: int = 2                      # Frames to run at startup for warmup
    inference_iou_threshold: float = 0.45       # Lower IoU threshold for tighter non-maximum suppression (NMS)
    track_history_len: int = 5                  # Sliding window size for track validation
    track_confirm_frames: int = 3               # Required detections in sliding window to confirm track
    track_max_ttl: int = 8                      # Frames to keep track after target goes out of sight


    # ── Performance ───────────────────────────────────────────────────────────
    target_fps: int = 30
    frame_skip_latency_ms: float = 50.0         # Skip frames if inference > this
    fps_window_size: int = 30                   # Sliding window for FPS average

    # ── Alert system ──────────────────────────────────────────────────────────
    alert_cooldown_sec: float = 3.0             # Min seconds between audio alerts
    alert_high_conf_threshold: float = 0.75     # Confidence above this → high-priority alert

    # ── Paths ─────────────────────────────────────────────────────────────────
    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent

    @property
    def logs_dir(self) -> Path:
        d = self.project_root / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def snapshots_dir(self) -> Path:
        d = self.logs_dir / "snapshots"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def recordings_dir(self) -> Path:
        d = self.logs_dir / "recordings"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def sounds_dir(self) -> Path:
        return self.project_root / "assets" / "sounds"

    @property
    def styles_dir(self) -> Path:
        return self.project_root / "assets" / "styles"

    @property
    def icons_dir(self) -> Path:
        return self.project_root / "assets" / "icons"

    # ── UI ────────────────────────────────────────────────────────────────────
    window_title: str = "AI Animal Roadway Detection System"
    min_width: int = 1280
    min_height: int = 820

    # ── Log table ─────────────────────────────────────────────────────────────
    log_max_rows: int = 100
    log_dedup_sec: float = 2.0                  # Suppress duplicate entries within this window


# Singleton instance — import and use `cfg` directly
cfg = Config()
