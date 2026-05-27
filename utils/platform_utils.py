"""
platform_utils.py — Cross-platform compatibility helpers.
Detects OS, selects correct camera backends, and queries GPU info.
"""

from __future__ import annotations
import platform
import sys
import cv2
from typing import Optional

OS = platform.system()   # 'Linux', 'Windows', 'Darwin'


def get_camera_backend() -> int:
    """Returns the correct cv2 VideoCapture backend for this OS."""
    if OS == "Windows":
        return cv2.CAP_DSHOW
    elif OS == "Linux":
        return cv2.CAP_V4L2
    elif OS == "Darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


def open_camera(index: int = 0) -> cv2.VideoCapture:
    """Open a camera with the OS-appropriate backend with fallback."""
    backend = get_camera_backend()
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened() and backend != cv2.CAP_ANY:
        cap.release()
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def enumerate_cameras(max_index: int = 4) -> list:
    """Return list of accessible camera indices 0..max_index-1."""
    available = []
    backend = get_camera_backend()
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            available.append(i)
        cap.release()
    return available


def get_cuda_info() -> dict:
    """Query torch CUDA availability and return info dict."""
    info = {
        "available": False,
        "device_name": "N/A",
        "memory_total_gb": 0.0,
        "memory_free_gb": 0.0,
        "cuda_version": "N/A",
    }
    try:
        import torch
        if torch.cuda.is_available():
            info["available"] = True
            info["device_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda or "unknown"
            props = torch.cuda.get_device_properties(0)
            info["memory_total_gb"] = props.total_memory / 1e9
            info["memory_free_gb"] = (
                props.total_memory - torch.cuda.memory_allocated(0)
            ) / 1e9
    except Exception:
        pass
    return info


def get_mps_available() -> bool:
    """True if Apple MPS (Metal) is available."""
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


def get_ui_font_family() -> str:
    """Best available system UI font for the current OS."""
    if OS == "Windows":
        return "Segoe UI"
    elif OS == "Darwin":
        return "-apple-system"
    return "Ubuntu, DejaVu Sans, Liberation Sans, sans-serif"
