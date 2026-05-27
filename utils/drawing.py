"""
drawing.py — OpenCV drawing helpers for the detection overlay.
All functions mutate the frame in-place for performance (no copies).
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple

Color = Tuple[int, int, int]   # BGR


# ──────────────────────────────────────────────────────────────────────────────
# Color palette
# ──────────────────────────────────────────────────────────────────────────────

HIGH_CONF_COLOR: Color  = (50,  220, 50)    # Vivid green  (BGR)
MED_CONF_COLOR: Color   = (0,   165, 255)   # Orange       (BGR)
LOW_CONF_COLOR: Color   = (60,  60,  220)   # Indigo-blue  (BGR)
LABEL_BG_ALPHA: float   = 0.65


def _conf_color(conf: float) -> Color:
    if conf >= 0.80:
        return HIGH_CONF_COLOR
    elif conf >= 0.55:
        return MED_CONF_COLOR
    return LOW_CONF_COLOR


# ──────────────────────────────────────────────────────────────────────────────
# Bounding Box with Corner Accents
# ──────────────────────────────────────────────────────────────────────────────

def draw_detection_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    conf: float,
    color: Color | None = None,
) -> None:
    """
    Draws a premium corner-accent bounding box + label badge.
    Operates in-place on `frame`.
    """
    if color is None:
        color = _conf_color(conf)

    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return

    # ── Main rectangle (semi-transparent via overlay trick) ──────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    # ── Solid border on top of overlay ───────────────────────────────────────
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    # ── Corner accents ────────────────────────────────────────────────────────
    lc = min(18, bw // 4, bh // 4)
    thickness = 3
    corners = [
        ((x1, y1),      (x1 + lc, y1),    (x1, y1 + lc)),
        ((x2, y1),      (x2 - lc, y1),    (x2, y1 + lc)),
        ((x1, y2),      (x1 + lc, y2),    (x1, y2 - lc)),
        ((x2, y2),      (x2 - lc, y2),    (x2, y2 - lc)),
    ]
    for corner, h_pt, v_pt in corners:
        cv2.line(frame, corner, h_pt, color, thickness)
        cv2.line(frame, corner, v_pt, color, thickness)

    # ── Label badge ───────────────────────────────────────────────────────────
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.50
    font_thick = 1
    text       = f"{label}  {conf:.0%}"
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thick)

    pad = 4
    bx1, by1 = x1, y1 - th - baseline - pad * 2
    bx2, by2 = x1 + tw + pad * 2, y1

    # Clamp badge inside frame
    if by1 < 0:
        by1 = y2
        by2 = y2 + th + baseline + pad * 2

    # Badge background (semi-transparent)
    badge_overlay = frame.copy()
    cv2.rectangle(badge_overlay, (bx1, by1), (bx2, by2), color, -1)
    cv2.addWeighted(badge_overlay, LABEL_BG_ALPHA, frame, 1 - LABEL_BG_ALPHA, 0, frame)

    # Badge text
    cv2.putText(
        frame, text,
        (bx1 + pad, by2 - baseline - pad // 2),
        font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA
    )


# ──────────────────────────────────────────────────────────────────────────────
# HUD Overlay (top-left status bar inside the video)
# ──────────────────────────────────────────────────────────────────────────────

def draw_hud_overlay(
    frame: np.ndarray,
    fps: float,
    det_count: int,
    mode_flags: dict[str, bool],
) -> None:
    """
    Draws a semi-transparent HUD in the top-left corner of the frame.
    mode_flags: {'thermal': bool, 'night': bool, 'boost': bool, 'edge': bool}
    """
    h, w = frame.shape[:2]
    hud_h = 34
    hud_w = 320

    # Background strip
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (hud_w, hud_h), (10, 14, 24), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # FPS
    fps_color = (50, 220, 80) if fps >= 25 else (0, 165, 255) if fps >= 15 else (60, 60, 220)
    cv2.putText(frame, f"FPS {fps:5.1f}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, fps_color, 1, cv2.LINE_AA)

    # Detection count
    cv2.putText(frame, f"| DET {det_count:02d}", (100, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    # Active mode indicator
    active_modes = [k.upper()[:3] for k, v in mode_flags.items() if v]
    if active_modes:
        mode_str = "| " + " ".join(active_modes)
        cv2.putText(frame, mode_str, (195, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 220), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# Animated scan-line (bonus)
# ──────────────────────────────────────────────────────────────────────────────

def draw_scan_line(frame: np.ndarray, tick: int) -> None:
    """Draws a moving horizontal scan line for a radar/sensor aesthetic."""
    h, w = frame.shape[:2]
    y = int((tick * 3) % h)
    overlay = frame.copy()
    cv2.line(overlay, (0, y), (w, y), (0, 255, 180), 1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
