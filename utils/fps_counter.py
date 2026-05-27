"""
fps_counter.py — Sliding-window FPS counter.
Thread-safe, no global state, zero dependencies beyond stdlib.
"""

from __future__ import annotations
import time
from collections import deque
from threading import Lock


class FPSCounter:
    """
    Maintains a deque of recent frame timestamps and computes a
    stable rolling-average FPS over the last `window` frames.
    """

    def __init__(self, window: int = 30) -> None:
        self._window = window
        self._times: deque[float] = deque(maxlen=window)
        self._lock = Lock()

    def tick(self) -> None:
        """Call once per frame to record a timestamp."""
        with self._lock:
            self._times.append(time.perf_counter())

    def get_fps(self) -> float:
        """Return current FPS as a float (0.0 if fewer than 2 ticks recorded)."""
        with self._lock:
            if len(self._times) < 2:
                return 0.0
            elapsed = self._times[-1] - self._times[0]
            if elapsed <= 0:
                return 0.0
            return (len(self._times) - 1) / elapsed

    def reset(self) -> None:
        with self._lock:
            self._times.clear()
