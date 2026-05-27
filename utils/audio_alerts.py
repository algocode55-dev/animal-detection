"""
audio_alerts.py — Cross-platform audio alert system.
Runs in a dedicated QThread to avoid blocking the GUI.

Backend priority (auto-detected):
  1. simpleaudio  — best quality, cross-platform
  2. pygame.mixer — widely available
  3. Qt QSoundEffect — pure Qt, no extra deps
  4. ASCII bell fallback — always works
"""

from __future__ import annotations
import io
import queue
import struct
import time
import math
import threading
from pathlib import Path

from PyQt6.QtCore import QThread


# ──────────────────────────────────────────────────────────────────────────────
# WAV synthesis (no file needed)
# ──────────────────────────────────────────────────────────────────────────────

def _synth_wav(freq: float = 880.0, duration: float = 0.18,
               sample_rate: int = 44100, amplitude: float = 0.4) -> bytes:
    """Synthesise a short sine-wave WAV in memory."""
    n_samples = int(sample_rate * duration)
    # Build PCM data (16-bit signed, mono)
    pcm = bytearray()
    for i in range(n_samples):
        t = i / sample_rate
        # Envelope: quick fade-in (5 ms) and fade-out (20 ms)
        env = 1.0
        if i < sample_rate * 0.005:
            env = i / (sample_rate * 0.005)
        elif i > n_samples - sample_rate * 0.02:
            env = (n_samples - i) / (sample_rate * 0.02)
        value = int(amplitude * env * 32767 * math.sin(2 * math.pi * freq * t))
        pcm += struct.pack('<h', max(-32768, min(32767, value)))

    data_size = len(pcm)
    # WAV header
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size,
        b'WAVE', b'fmt ', 16,
        1,             # PCM
        1,             # mono
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b'data', data_size
    )
    return header + bytes(pcm)


# ──────────────────────────────────────────────────────────────────────────────
# Alert thread
# ──────────────────────────────────────────────────────────────────────────────

class AlertThread(QThread):
    """
    Dedicated QThread for non-blocking audio alert playback.
    Call trigger() from any thread to enqueue an alert.
    """

    PRIORITY_LOW  = 1
    PRIORITY_HIGH = 2

    def __init__(self, cooldown_sec: float = 3.0, parent=None):
        super().__init__(parent)
        self._q: queue.Queue[int] = queue.Queue(maxsize=4)
        self._cooldown = cooldown_sec
        self._last_played: float = 0.0
        self._running = True
        self._backend: str = self._detect_backend()
        self._wav_cache: dict[int, bytes] = {
            self.PRIORITY_LOW:  _synth_wav(freq=660.0, duration=0.15),
            self.PRIORITY_HIGH: _synth_wav(freq=1100.0, duration=0.22),
        }

    # ── Backend detection ─────────────────────────────────────────────────────

    @staticmethod
    def _detect_backend() -> str:
        try:
            import simpleaudio  # noqa: F401
            return "simpleaudio"
        except ImportError:
            pass
        try:
            import pygame.mixer  # noqa: F401
            return "pygame"
        except ImportError:
            pass
        try:
            from PyQt6.QtMultimedia import QSoundEffect  # noqa: F401
            return "qt"
        except ImportError:
            pass
        return "bell"

    # ── Public API ────────────────────────────────────────────────────────────

    def trigger(self, priority: int = PRIORITY_LOW) -> None:
        """Thread-safe: enqueue an alert (drops if queue full or in cooldown)."""
        now = time.time()
        if now - self._last_played < self._cooldown:
            return
        try:
            self._q.put_nowait(priority)
        except queue.Full:
            pass

    def stop(self) -> None:
        self._running = False
        try:
            self._q.put_nowait(0)   # Unblock the wait
        except queue.Full:
            pass
        self.wait(2000)

    # ── Thread body ───────────────────────────────────────────────────────────

    def run(self) -> None:
        while self._running:
            try:
                priority = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            if not self._running or priority == 0:
                break
            self._last_played = time.time()
            self._play(priority)

    def _play(self, priority: int) -> None:
        wav = self._wav_cache.get(priority, self._wav_cache[self.PRIORITY_LOW])
        try:
            if self._backend == "simpleaudio":
                import simpleaudio as sa
                import numpy as np
                audio = np.frombuffer(wav[44:], dtype=np.int16)   # skip WAV header
                wave_obj = sa.WaveObject(audio, 1, 2, 44100)
                play_obj = wave_obj.play()
                play_obj.wait_done()
                return

            if self._backend == "pygame":
                import pygame.mixer as mixer
                import numpy as np
                if not mixer.get_init():
                    mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
                audio = np.frombuffer(wav[44:], dtype=np.int16)
                sound = mixer.Sound(audio)
                sound.play()
                time.sleep(0.25)
                return

            if self._backend == "qt":
                # Write tmp WAV and play via QSoundEffect
                import tempfile, os
                from PyQt6.QtMultimedia import QSoundEffect
                from PyQt6.QtCore import QUrl
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.write(wav)
                tmp.close()
                effect = QSoundEffect()
                effect.setSource(QUrl.fromLocalFile(tmp.name))
                effect.play()
                time.sleep(0.30)
                os.unlink(tmp.name)
                return

        except Exception:
            pass

        # Fallback: ASCII bell
        print('\a', end='', flush=True)
