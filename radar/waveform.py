"""Transmit side: PN sequence, RRC pulse and preamble (ported from the MATLAB link),
plus the periodic OFDM sounding symbol and an optional breathing-pace cue."""

import numpy as np

from .config import SounderConfig

GRAY_QPSK = np.array([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j]) / np.sqrt(2)


def lfsr_framesync(nbits: int) -> np.ndarray:
    """Order-11 LFSR, feedback x^11 + x^2 + 1, all-ones seed (port of lfsr_framesync.m)."""
    state = [1] * 11
    out = np.zeros(max(nbits, 0), dtype=np.int8)
    for k in range(len(out)):
        out[k] = state[-1]
        feedback = state[-1] ^ state[1]
        state = [feedback] + state[:-1]
    return out


def rrc(os_factor: int, rolloff: float, filterlength: int) -> np.ndarray:
    """Root-raised-cosine taps with unit energy, 2*filterlength+1 taps (port of rrc.m)."""
    n = np.arange(-filterlength, filterlength + 1) / os_factor
    if rolloff == 0:
        h = np.sinc(n)
    else:
        if float(os_factor / (4 * rolloff)).is_integer():
            raise ValueError("os_factor/(4*rolloff_factor) must not be an integer.")
        h = (4 * rolloff / np.pi * np.cos((1 + rolloff) * np.pi * n)
             + (1 - rolloff) * np.sinc((1 - rolloff) * n)) / (1 - (4 * rolloff * n) ** 2)
    return h / np.sqrt(np.sum(np.abs(h) ** 2))


def rrc_pulse(cfg: SounderConfig) -> np.ndarray:
    return rrc(cfg.sc_os, cfg.rrc_rolloff, cfg.rrc_span_syms * cfg.sc_os)


def preamble_baseband(cfg: SounderConfig) -> np.ndarray:
    """BPSK preamble shaped with the RRC pulse, unit average power (txofdm.m step 6)."""
    bits = lfsr_framesync(cfg.sc_nsyms)
    bpsk = 1.0 - 2.0 * bits
    ups = np.zeros(len(bpsk) * cfg.sc_os)
    ups[:: cfg.sc_os] = bpsk
    pulse = rrc_pulse(cfg)
    half = cfg.rrc_span_syms * cfg.sc_os
    shaped = np.convolve(ups, pulse, mode="full")[half:-half]
    return shaped / np.sqrt(np.mean(np.abs(shaped) ** 2))


def sounding_phases(cfg: SounderConfig) -> np.ndarray:
    """Unit-modulus subcarrier values of the sounding symbol."""
    k = np.arange(cfg.n_sub)
    if cfg.phase_scheme == "newman":
        # Quadratic (Newman) phases: flat spectrum with a crest factor close to a chirp.
        return np.exp(1j * np.pi * k ** 2 / cfg.n_sub)
    if cfg.phase_scheme == "pn":
        # QPSK from the LFSR bits that follow the preamble, Gray-mapped as in the link.
        bits = lfsr_framesync(cfg.sc_nsyms + 2 * cfg.n_sub)[cfg.sc_nsyms:].reshape(-1, 2)
        return GRAY_QPSK[2 * bits[:, 0] + bits[:, 1]]
    raise ValueError(f"unknown phase scheme {cfg.phase_scheme!r}")


def sounding_symbol(cfg: SounderConfig) -> np.ndarray:
    """One period (cfg.period samples) of the real passband sounding signal, peak 1.

    Subcarrier k sits exactly on FFT bin k0 + k of a period-long window, so repeating the
    symbol makes every window of the recording a cyclic copy: no cyclic prefix is needed.
    """
    P, k0 = cfg.period, cfg.k0
    bins = k0 + np.arange(cfg.n_sub)
    if bins[-1] >= P // 2:
        raise ValueError("sounding band exceeds Nyquist")
    spec = np.zeros(P, dtype=complex)
    spec[bins] = sounding_phases(cfg)
    spec[P - bins] = np.conj(spec[bins])
    x = np.fft.ifft(spec).real
    return x / np.max(np.abs(x))


def reference_spectrum(cfg: SounderConfig) -> np.ndarray:
    """FFT of one transmitted period on the sounding bins (the LS reference)."""
    return np.fft.fft(sounding_symbol(cfg))[cfg.k0: cfg.k0 + cfg.n_sub]


def crest_factor_db(x: np.ndarray) -> float:
    return 20 * np.log10(np.max(np.abs(x)) / np.sqrt(np.mean(x ** 2)))


def pacing_cue(n: int, fs: int, bpm: float, level: float = 0.06,
               f_lo: float = 440.0, f_hi: float = 660.0) -> np.ndarray:
    """Breathing-pace tone: pitch rises while inhaling, falls while exhaling.

    It sits far below the 6-10 kHz sounding band, so the receiver never sees it. It sets the
    rate the person tries to follow: a reference, not exact ground truth, since nobody
    breathes exactly on the tone.
    """
    t = np.arange(n) / fs
    cycle = (t * bpm / 60.0) % 1.0
    rise = np.where(cycle < 0.5, cycle * 2, 2 - cycle * 2)      # triangle 0 -> 1 -> 0
    inst_f = f_lo + (f_hi - f_lo) * rise
    phase = 2 * np.pi * np.cumsum(inst_f) / fs
    return level * np.sin(phase)


def _fade(n: int, fade: int) -> np.ndarray:
    w = np.ones(n)
    if fade > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
        w[:fade] = ramp
        w[-fade:] = ramp[::-1]
    return w


def build_tx(cfg: SounderConfig, seconds: float, cue_bpm: float | None = None):
    """Full transmit waveform: [silence | preamble | periodic sounding train | silence].

    Returns (tx, layout) where layout holds the sample indices the receiver needs.
    """
    fs = cfg.fs
    lead = int(round(cfg.lead_s * fs))
    tail = int(round(cfg.tail_s * fs))

    pre = preamble_baseband(cfg)
    n = np.arange(len(pre))
    pre_pb = pre * np.cos(2 * np.pi * cfg.f_c / fs * n)
    pre_pb /= np.max(np.abs(pre_pb))

    n_periods = int(seconds * cfg.frame_rate)
    train = np.tile(sounding_symbol(cfg), n_periods)
    train *= _fade(len(train), int(round(cfg.fade_s * fs)))
    if cue_bpm:
        train = train + pacing_cue(len(train), fs, cue_bpm, level=0.15)

    tx = np.concatenate([np.zeros(lead), pre_pb, train, np.zeros(tail)])
    tx = cfg.amplitude * tx / np.max(np.abs(tx))
    layout = {
        "lead": lead,
        "preamble_len": len(pre_pb),
        "sound_start": lead + len(pre_pb),
        "n_periods": n_periods,
        "period": cfg.period,
        "cue_bpm": cue_bpm,
        "total": len(tx),
    }
    return tx.astype(np.float32), layout
