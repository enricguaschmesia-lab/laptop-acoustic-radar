"""Receive side: LS channel estimation per sounding period, impulse responses, static
clutter removal, range-Doppler maps and breathing-rate estimation."""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import detrend

from .config import SPEED_OF_SOUND, SounderConfig
from .sync import find_sounding_start
from .waveform import reference_spectrum

GUARD_TAPS = 4  # taps kept before the direct path after alignment
# Arc demodulation is skipped when the I/Q points' distance to the fitted centre varies by
# more than this (clearly not a circle). Real breathing bins ranged 0.21-0.48; the
# survivors are then vetted by spectral purity and voting in breathing_rate.
MAX_ARC_CV = 0.6
MIN_PURITY = 0.3        # fraction of a candidate's spectrum inside the breathing band
VOTE_TOL_BPM = 1.5      # candidates within this of each other agree
VOTE_TAPS = 4           # range bins (by breathing-band energy) that vote


def channel_estimates(rx: np.ndarray, start: int, cfg: SounderConfig, n_periods: int,
                      skip_s: float = 0.3, tail_periods: int = 1) -> np.ndarray:
    """Per-period, per-subcarrier LS estimates H[t, k] = Y_t[k0 + k] / X[k].

    The first skip_s seconds are dropped: they still contain the echo of the preamble.
    """
    P = cfg.period
    skip = int(np.ceil(skip_s * cfg.frame_rate))
    n = min(n_periods - skip - tail_periods, (len(rx) - start) // P - skip)
    if n <= 0:
        raise ValueError("recording too short for the requested sounding train")
    first = start + skip * P
    frames = rx[first: first + n * P].reshape(n, P)
    Y = np.fft.fft(frames, axis=1)[:, cfg.k0: cfg.k0 + cfg.n_sub]
    return Y / reference_spectrum(cfg)[None, :]


def impulse_response(H: np.ndarray) -> np.ndarray:
    """Complex baseband impulse response per period, tap spacing 1/B."""
    return np.fft.ifft(H, axis=1)


def align_direct_path(h: np.ndarray, guard: int = GUARD_TAPS):
    """Circularly shift taps so the strongest (direct speaker->mic) path sits at `guard`."""
    direct = int(np.argmax(np.mean(np.abs(h) ** 2, axis=0)))
    return np.roll(h, guard - direct, axis=1), direct


def range_axis(cfg: SounderConfig, guard: int = GUARD_TAPS) -> np.ndarray:
    """Approximate reflector range (m) per tap: half the extra path over the direct path."""
    return (np.arange(cfg.n_sub) - guard) * cfg.range_bin


def remove_static(h: np.ndarray, frame_rate: float, window_s: float | None = None) -> np.ndarray:
    """Remove the static clutter (direct path, furniture): subtract the slow-time mean,
    or a moving average over window_s seconds when given (a high-pass in slow time)."""
    if window_s is None:
        return h - h.mean(axis=0, keepdims=True)
    w = max(1, int(round(window_s * frame_rate)))
    smooth = (uniform_filter1d(h.real, w, axis=0, mode="nearest")
              + 1j * uniform_filter1d(h.imag, w, axis=0, mode="nearest"))
    return h - smooth


def range_doppler(h_dyn: np.ndarray, cfg: SounderConfig, win: int = 128, hop: int = 32):
    """Short-time FFT over slow time. Returns (maps[n_win, doppler, range], doppler_hz, t_s)."""
    n = (len(h_dyn) - win) // hop + 1
    if n <= 0:
        raise ValueError("not enough periods for one Doppler window")
    taper = np.hanning(win)[:, None]
    maps = np.stack([np.fft.fftshift(np.fft.fft(h_dyn[i * hop: i * hop + win] * taper, axis=0), axes=0)
                     for i in range(n)])
    doppler = np.fft.fftshift(np.fft.fftfreq(win, 1 / cfg.frame_rate))
    t = (np.arange(n) * hop + win / 2) / cfg.frame_rate
    return np.abs(maps) ** 2, doppler, t


def doppler_to_speed(f_d, cfg: SounderConfig):
    return np.asarray(f_d) * SPEED_OF_SOUND / (2 * cfg.f_center)


def pca_projection(z: np.ndarray) -> np.ndarray:
    """Project a complex slow-time series onto its principal I/Q axis.

    A breathing chest traces a short arc in the I/Q plane (5 mm is ~85 deg of phase at
    8 kHz). Projecting on the arc's main axis gives a signal close to linear in
    displacement without needing the arc centre.
    """
    pts = np.column_stack([z.real - z.real.mean(), z.imag - z.imag.mean()])
    _, vecs = np.linalg.eigh(pts.T @ pts)
    return pts @ vecs[:, -1]


def arc_phase(z: np.ndarray) -> np.ndarray | None:
    """Unwrapped phase of a slow-time I/Q series around its fitted arc centre
    (arctangent demodulation with DC-offset compensation, Kasa circle fit).

    Needed when the reflector moves more than ~lambda/4 (1 cm at 8 kHz): the I/Q points
    then wrap around most of a circle and a straight-line projection folds each breath
    into two humps, doubling the estimated rate. Returns None, so the caller falls back
    to the projection, when the points lie too close to a straight line for a stable fit
    (small motion, where the projection is fine) or are not on a circle at all: if their
    distance to the fitted centre varies by more than MAX_ARC_CV, the trajectory passes
    near the centre and unwrapping slips by whole turns.
    """
    x, y = z.real, z.imag
    A = np.column_stack([x, y, np.ones_like(x)])
    D, E, F = np.linalg.lstsq(A, -(x ** 2 + y ** 2), rcond=None)[0]
    xc, yc = -D / 2, -E / 2
    r2 = xc ** 2 + yc ** 2 - F
    spread = np.sqrt(np.var(x) + np.var(y))
    if not np.isfinite(r2) or r2 <= 0 or np.sqrt(r2) > 20 * spread:
        return None
    w = (x - xc) + 1j * (y - yc)
    dist = np.abs(w)
    if dist.std() / dist.mean() > MAX_ARC_CV:
        return None
    return np.unwrap(np.angle(w))


def _spectrum(x: np.ndarray, fs: float, nfft: int = 1 << 16):
    x = detrend(x) * np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x, nfft)) ** 2
    return np.fft.rfftfreq(nfft, 1 / fs), spec


def _peak_freq(f: np.ndarray, spec: np.ndarray, band) -> float:
    idx = np.flatnonzero((f >= band[0]) & (f <= band[1]))
    i = idx[np.argmax(spec[idx])]
    if 0 < i < len(spec) - 1:   # parabolic interpolation of the peak
        a, b, c = spec[i - 1], spec[i], spec[i + 1]
        denom = a - 2 * b + c
        if denom != 0:
            return float(f[i] + 0.5 * (a - c) / denom * (f[1] - f[0]))
    return float(f[i])


def breathing_rate(h: np.ndarray, cfg: SounderConfig, guard: int = GUARD_TAPS,
                   min_range: float = 0.15, max_range: float = 1.5,
                   band=(0.1, 0.7), tap: int | None = None) -> dict:
    """Breathing rate from the slow-time signals of the chest range bins.

    The VOTE_TAPS bins with the most breathing-band energy between min_range and max_range
    (or only `tap`, when given) each yield up to two candidates: the arc-phase displacement
    (see arc_phase) and the principal-axis projection. Candidates with less than
    MIN_PURITY of their spectrum in the band are dropped, the rest are grouped by rate, and
    the group with the most purity wins. Folding can only double a rate, never halve it,
    so when a well-supported group sits at half the winner's rate, that group wins. The
    rate is the median of the winning group; tap, method and signal are its purest member.

    Frozen on 2026-10-20 after tuning on sessions 1-2 and the go/no-go capture; sessions
    3-4 are the held-out test.
    """
    r = range_axis(cfg, guard)
    in_range = np.flatnonzero((r >= min_range) & (r <= max_range))
    scores = []
    for k in in_range:
        f, spec = _spectrum(pca_projection(h[:, k]), cfg.frame_rate, nfft=1 << 14)
        in_band = (f >= band[0]) & (f <= band[1])
        scores.append(spec[in_band].sum())
    scores = np.asarray(scores)
    taps = [tap] if tap is not None else [int(k) for k in in_range[np.argsort(scores)[::-1][:VOTE_TAPS]]]

    cands = []
    for k in taps:
        z = h[:, k]
        signals = [("pca", pca_projection(z))]
        ph = arc_phase(z)
        if ph is not None:
            signals.append(("arc", ph * cfg.wavelength / (4 * np.pi)))
        for method, s in signals:
            f, spec = _spectrum(s, cfg.frame_rate)
            in_band = (f >= band[0]) & (f <= band[1])
            cands.append({"tap": k, "method": method, "signal": s, "freqs": f, "spectrum": spec,
                          "rate_bpm": 60 * _peak_freq(f, spec, band),
                          "band_fraction": float(spec[in_band].sum() / spec[1:].sum())})
    good = [c for c in cands if c["band_fraction"] >= MIN_PURITY] or cands

    groups = []                                  # greedy clustering by rate
    for c in sorted(good, key=lambda c: c["rate_bpm"]):
        if groups and c["rate_bpm"] - groups[-1][-1]["rate_bpm"] <= VOTE_TOL_BPM:
            groups[-1].append(c)
        else:
            groups.append([c])
    weight = [sum(c["band_fraction"] for c in g) for g in groups]
    centre = [np.median([c["rate_bpm"] for c in g]) for g in groups]
    win = int(np.argmax(weight))
    for i in range(len(groups)):
        if abs(2 * centre[i] - centre[win]) <= 2 * VOTE_TOL_BPM and weight[i] >= 0.5 * weight[win]:
            win = i
            break
    best = max(groups[win], key=lambda c: c["band_fraction"])
    rate = float(np.median([c["rate_bpm"] for c in groups[win]]))
    return {**best, "rate_bpm": rate, "range_m": float(r[best["tap"]]), "candidates": [
                {k: c[k] for k in ("tap", "method", "rate_bpm", "band_fraction")} for c in cands],
            "tap_scores": dict(zip(in_range.tolist(), scores.tolist()))}


def motion_energy(h_dyn: np.ndarray, cfg: SounderConfig, guard: int = GUARD_TAPS,
                  min_range: float = 0.15, max_range: float = 2.0, window: int = 64) -> np.ndarray:
    """Energy of the clutter-removed response in a range band, per window of periods."""
    r = range_axis(cfg, guard)
    sel = (r >= min_range) & (r <= max_range)
    p = np.sum(np.abs(h_dyn[:, sel]) ** 2, axis=1)
    n = len(p) // window
    return p[: n * window].reshape(n, window).mean(axis=1)


def match_static(H_ref: np.ndarray, H: np.ndarray, max_shift: float = 1.0, steps: int = 201) -> np.ndarray:
    """Align a mean channel H[k] to H_ref[k] with a fractional delay and complex gain.

    Captures taken minutes apart differ by sub-tap timing and level; without this the
    direct path leaks into the difference H - H_ref and hides weak static reflectors.
    The fit is dominated by the strong static paths, which is what we want to cancel.
    """
    k = np.arange(len(H))
    best, best_err = H, np.inf
    for d in np.linspace(-max_shift, max_shift, steps):
        Hd = H * np.exp(2j * np.pi * k * d / len(H))
        g = np.vdot(Hd, H_ref) / np.vdot(Hd, Hd)
        err = np.sum(np.abs(g * Hd - H_ref) ** 2)
        if err < best_err:
            best, best_err = g * Hd, err
    return best


def static_difference(H_empty: np.ndarray, H_obj: np.ndarray, cfg: SounderConfig,
                      guard: int = GUARD_TAPS) -> np.ndarray:
    """|h_obj - h_empty| per tap from the time-averaged channels of two captures."""
    ref = H_empty.mean(axis=0)
    obj = match_static(ref, H_obj.mean(axis=0))
    h_ref, direct = align_direct_path(impulse_response(ref[None, :]), guard)
    h_obj = np.roll(impulse_response(obj[None, :]), guard - direct, axis=1)
    return np.abs(h_obj - h_ref)[0]


def process_recording(rx: np.ndarray, cfg: SounderConfig, layout: dict, skip_s: float = 0.3) -> dict:
    """Sync, estimate and align: recording -> aligned impulse responses h[t, tap]."""
    start, sync_dbg = find_sounding_start(rx, cfg)
    H = channel_estimates(rx, start, cfg, layout["n_periods"], skip_s=skip_s)
    h, direct = align_direct_path(impulse_response(H))
    return {"H": H, "h": h, "start": start, "direct_tap_raw": direct,
            "latency_samples": start - layout["sound_start"], "sync": sync_dbg}
