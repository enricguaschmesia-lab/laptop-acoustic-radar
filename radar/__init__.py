"""Monostatic acoustic OFDM radar on a laptop's own speaker and microphone.

The transmit chain reuses the EE-442 acoustic OFDM link (preamble, pulse shaping, frame
sync); the data symbols are replaced by a periodic OFDM sounding symbol, so every 16 ms of
the recording yields one channel impulse response.
"""

from .config import SounderConfig

__all__ = ["SounderConfig"]
__version__ = "1.0.0"
