"""Frame synchronisation, ported from the MATLAB link (rxofdm.m step 3 and frame_sync.m)."""

import numpy as np
from scipy.signal import correlate, fftconvolve

from .config import SounderConfig
from .waveform import lfsr_framesync, rrc_pulse


def downconvert(x: np.ndarray, fs: float, fc: float) -> np.ndarray:
    n = np.arange(len(x))
    return x * np.exp(-2j * np.pi * fc / fs * n)


def matched_filter(x_bb: np.ndarray, pulse: np.ndarray) -> np.ndarray:
    """Equivalent of MATLAB conv(x, conj(flipud(pulse)), 'same') for an odd-length pulse."""
    return fftconvolve(x_bb, np.conj(pulse[::-1]), mode="same")


def frame_sync(mf: np.ndarray, nsyms: int, os_factor: int):
    """Normalised correlation T(p) = |s^T r_p|^2 / (r_p^H r_p) against the BPSK preamble.

    r_p takes one sample per symbol ending one symbol before p. Returns the 0-based index
    p of the first sample after the preamble (MATLAB's beginning_of_data minus one) and
    a debug dict mirroring fs_debug.
    """
    seq = 1.0 - 2.0 * lfsr_framesync(nsyms)
    L = os_factor
    span = L * nsyms
    num_positions = len(mf) - span
    if num_positions <= 0:
        raise ValueError("received signal is shorter than the preamble")

    kernel = np.zeros(L * (nsyms - 1) + 1)
    kernel[::L] = seq
    c = correlate(mf, kernel, mode="valid", method="fft")[:num_positions]
    energy = correlate(np.abs(mf) ** 2, np.abs(kernel), mode="valid", method="fft")[:num_positions]
    energy = np.maximum(energy, 0.0)        # FFT round-off can dip below zero
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.where(energy > np.finfo(float).eps, np.abs(c) ** 2 / energy, 0.0)

    idx_peak = int(np.argmax(T))
    peak = float(T[idx_peak])
    noise_len = max(10, int(np.floor(0.2 * num_positions + 0.5)))
    noise = T[:noise_len]
    threshold = max(noise.mean() + 8 * noise.std(ddof=1), 0.4 * peak)
    debug = {"T_vals": T, "p_start": span, "idx_peak": idx_peak,
             "peak_val": peak, "threshold": float(threshold)}
    return span + idx_peak, debug


def find_sounding_start(rx: np.ndarray, cfg: SounderConfig):
    """Locate the first sample of the sounding train in a real passband recording."""
    pulse = rrc_pulse(cfg)
    mf = matched_filter(downconvert(rx, cfg.fs, cfg.f_c), pulse)
    return frame_sync(mf, cfg.sc_nsyms, cfg.sc_os)
