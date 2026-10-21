"""Synthetic acoustic channel for testing the receiver without hardware."""

from collections.abc import Callable, Sequence

import numpy as np
from scipy.signal import resample_poly

from .config import SPEED_OF_SOUND

Delay = float | Callable[[np.ndarray], np.ndarray]


def propagate(tx: np.ndarray, fs: int, paths: Sequence[tuple[float, Delay]],
              noise_std: float = 0.0, seed: int = 0, oversample: int = 8,
              chunk_s: float = 2.0) -> np.ndarray:
    """Sum of delayed, scaled copies of tx plus white noise.

    paths: (gain, delay) pairs; delay is seconds, constant or a function of time (s) for
    moving reflectors (it must be a pure function of t: it is evaluated chunk by chunk).
    Fractional delays use linear interpolation on an 8x oversampled copy of tx (error
    around -60 dB at 10 kHz). Output is built in chunks so long captures fit in memory.
    """
    tx = tx.astype(float)
    n = len(tx)
    y = np.zeros(n)
    chunk = max(1, int(chunk_s * fs))
    margin = 64                       # input samples, covers the resampling filter
    for a in range(0, n, chunk):
        t = np.arange(a, min(n, a + chunk)) / fs
        taus = [(g, d(t) if callable(d) else np.full_like(t, float(d))) for g, d in paths]
        lo = min(tau.min() for _, tau in taus)
        hi = max(tau.max() for _, tau in taus)
        i0 = max(0, int(np.floor((t[0] - hi) * fs)) - margin)
        i1 = min(n, int(np.ceil((t[-1] - lo) * fs)) + margin + 1)
        if i1 <= i0:
            continue
        fine = resample_poly(tx[i0:i1], oversample, 1)
        grid = i0 + np.arange(len(fine)) / oversample          # in input-sample units
        for gain, tau in taus:
            y[a: a + len(t)] += gain * np.interp((t - tau) * fs, grid, fine, left=0.0, right=0.0)
    rng = np.random.default_rng(seed)
    return y + noise_std * rng.standard_normal(n)


def reflector_delay(range_m, latency_s: float = 0.0):
    """Monostatic round-trip delay for a reflector at range_m (constant or callable)."""
    if callable(range_m):
        return lambda t: latency_s + 2 * range_m(t) / SPEED_OF_SOUND
    return latency_s + 2 * range_m / SPEED_OF_SOUND


def breathing_range(base_m: float, amplitude_m: float, bpm: float):
    """Chest range over time for sinusoidal breathing."""
    return lambda t: base_m + amplitude_m * np.sin(2 * np.pi * bpm / 60 * t)
