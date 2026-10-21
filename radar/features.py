"""Windowed inputs for the classifiers: range-Doppler and range-time maps for the CNN, and
hand-crafted features for the classical baselines.

A window is WIN periods (4.1 s) of the aligned impulse responses of both mics. Static
clutter is removed per window by subtracting the window mean, so a window needs nothing
outside itself (deployable as-is), and powers are normalised by the window's static power
to cancel gain differences between mics and sessions.
"""

import numpy as np

from .config import SPEED_OF_SOUND, SounderConfig
from .processing import GUARD_TAPS, range_axis

WIN = 256             # periods per window (4.1 s at 62.5 Hz)
HOP = 64              # periods between windows (1.0 s)
POOL = 2              # Doppler bins / frames pooled pairwise: 256 -> 128
MOTION_RANGE = (0.15, 2.5)
FAST_SPEED = 0.1      # m/s, threshold of the "fast motion" fraction
SLOW_DOPPLER = 1.0    # Hz, |f_d| below this counts as slow (breathing, sway)

FEATURE_NAMES = ("motion_db", "range_peak_m", "range_spread_m", "range_extent_m", "mean_speed",
                 "speed_spread", "fast_fraction", "slow_fraction", "doppler_asymmetry",
                 "burstiness")


def window_starts(n_frames: int, win: int = WIN, hop: int = HOP) -> range:
    return range(0, n_frames - win + 1, hop)


def _split(hw: np.ndarray):
    """Window -> (dynamic part restricted to range taps, static power)."""
    static = hw.mean(axis=0)
    dyn = (hw - static)[:, GUARD_TAPS:]
    return dyn, float(np.sum(np.abs(static) ** 2))


def _range_doppler(dyn: np.ndarray) -> np.ndarray:
    taper = np.hanning(len(dyn))[:, None]
    return np.abs(np.fft.fftshift(np.fft.fft(dyn * taper, axis=0), axes=0)) ** 2


def doppler_axis(cfg: SounderConfig, n: int = WIN) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftfreq(n, 1 / cfg.frame_rate))


def _pool(x: np.ndarray, k: int = POOL) -> np.ndarray:
    n = (len(x) // k) * k
    return x[:n].reshape(n // k, k, *x.shape[1:]).mean(axis=1)


def window_maps(hw_list: list[np.ndarray]) -> np.ndarray:
    """(2 * n_mics, 128, n_range) log-power maps: range-Doppler per mic, then range-time per mic."""
    rd, rt = [], []
    for hw in hw_list:
        dyn, static_power = _split(hw)
        norm = static_power + 1e-20
        rd.append(10 * np.log10(_pool(_range_doppler(dyn)) / norm + 1e-12))
        rt.append(10 * np.log10(_pool(np.abs(dyn) ** 2) / norm + 1e-12))
    return np.stack(rd + rt).astype(np.float32)


def window_features(hw_list: list[np.ndarray], cfg: SounderConfig) -> np.ndarray:
    """Hand-crafted features (FEATURE_NAMES), averaged over mics."""
    r = range_axis(cfg)[GUARD_TAPS:]
    sel = (r >= MOTION_RANGE[0]) & (r <= MOTION_RANGE[1])
    fd = doppler_axis(cfg, len(hw_list[0]))
    v = fd * SPEED_OF_SOUND / (2 * cfg.f_center)
    feats = []
    for hw in hw_list:
        dyn, static_power = _split(hw)
        p_rt = np.abs(dyn[:, sel]) ** 2                     # time x range
        p_range = p_rt.mean(axis=0)
        total = p_range.sum() + 1e-20
        rr = r[sel]
        r_mean = np.sum(p_range * rr) / total
        p_v = _range_doppler(dyn)[:, sel].sum(axis=1)        # power per Doppler bin
        pv_total = p_v.sum() + 1e-20
        per_frame = p_rt.sum(axis=1)
        feats.append([
            10 * np.log10(total / (static_power + 1e-20)),
            rr[np.argmax(p_range)],
            np.sqrt(np.sum(p_range * (rr - r_mean) ** 2) / total),
            np.sum(p_range > 0.1 * p_range.max()) * cfg.range_bin,
            np.sum(np.abs(v) * p_v) / pv_total,
            np.sqrt(np.sum(v ** 2 * p_v) / pv_total),
            p_v[np.abs(v) > FAST_SPEED].sum() / pv_total,
            p_v[np.abs(fd) < SLOW_DOPPLER].sum() / pv_total,
            abs(p_v[v > 0].sum() - p_v[v < 0].sum()) / pv_total,
            per_frame.std() / (per_frame.mean() + 1e-20),
        ])
    return np.mean(np.asarray(feats), axis=0).astype(np.float32)
