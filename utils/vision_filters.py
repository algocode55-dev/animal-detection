"""
vision_filters.py — Stateless OpenCV vision enhancement filters.
All functions take a BGR numpy array and return a new BGR numpy array.
Filters are designed to be fast and composable.
"""

from __future__ import annotations
import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Thermal Vision
# ──────────────────────────────────────────────────────────────────────────────

def apply_thermal(frame: np.ndarray, blend: float = 0.82) -> np.ndarray:
    """
    Simulated IR thermal vision using CLAHE equalization + JET colormap.
    blend: weight of the thermal layer (0.0 = original, 1.0 = pure thermal).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # CLAHE improves local contrast to simulate heat-gradient differentiation
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    thermal = cv2.applyColorMap(eq, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 1.0 - blend, thermal, blend, 0)


# ──────────────────────────────────────────────────────────────────────────────
# Night Vision
# ──────────────────────────────────────────────────────────────────────────────

def apply_night_vision(frame: np.ndarray, scanlines: bool = True) -> np.ndarray:
    """
    Simulated Gen-2 night-vision goggles effect:
    • CLAHE enhancement
    • Strong green phosphor channel
    • Slight vignette
    • Optional scanline overlay
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    out = np.zeros_like(frame)
    out[:, :, 1] = enhanced                                 # Green channel dominant
    out[:, :, 0] = (enhanced * 0.12).astype(np.uint8)      # Slight blue
    out[:, :, 2] = (enhanced * 0.08).astype(np.uint8)      # Barely-there red

    # Scanline overlay (every other horizontal line slightly darker)
    if scanlines:
        out[::2, :] = (out[::2, :] * 0.65).astype(np.uint8)

    # Vignette
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_dist = np.sqrt(cx**2 + cy**2)
    vignette = np.clip(1.0 - (dist / max_dist) ** 1.5, 0, 1).astype(np.float32)
    out = (out * vignette[:, :, np.newaxis]).astype(np.uint8)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Contrast / Brightness Boost
# ──────────────────────────────────────────────────────────────────────────────

def apply_contrast_boost(
    frame: np.ndarray, alpha: float = 1.35, beta: int = 20
) -> np.ndarray:
    """
    Linear contrast (alpha) and brightness (beta) adjustment.
    alpha > 1 increases contrast; beta > 0 increases brightness.
    """
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


# ──────────────────────────────────────────────────────────────────────────────
# Edge Enhance (bonus)
# ──────────────────────────────────────────────────────────────────────────────

_SHARPEN_KERNEL = np.array(
    [[-1, -1, -1],
     [-1,  9, -1],
     [-1, -1, -1]], dtype=np.float32
)


def apply_edge_enhance(frame: np.ndarray) -> np.ndarray:
    """Laplacian-based sharpening filter for detail enhancement."""
    return cv2.filter2D(frame, -1, _SHARPEN_KERNEL)
