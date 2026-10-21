"""Receiver checks on a synthetic monostatic channel with known reflectors."""

import numpy as np
import pytest

from radar.config import SPEED_OF_SOUND, SounderConfig
from radar.processing import GUARD_TAPS, breathing_rate, process_recording, range_axis, remove_static
from radar.sim import breathing_range, propagate, reflector_delay
from radar.waveform import build_tx, crest_factor_db, sounding_symbol

CFG = SounderConfig()
LATENCY = 0.023          # s, audio stack latency
D_SM = 0.15              # m, speaker -> mic distance (direct path)
NOISE = 1e-3


def clutter_paths():
    return [(1.0, LATENCY + D_SM / SPEED_OF_SOUND),         # direct speaker -> mic
            (0.3, reflector_delay(0.25, LATENCY)),            # table
            (0.1, reflector_delay(1.8, LATENCY))]             # wall


def expected_range(r):
    """range_axis measures half the extra path over the direct path."""
    return r - D_SM / 2


def run(seconds, extra_paths=(), cue_bpm=None, seed=0):
    tx, layout = build_tx(CFG, seconds, cue_bpm=cue_bpm)
    rx = propagate(tx, CFG.fs, clutter_paths() + list(extra_paths), noise_std=NOISE, seed=seed)
    return process_recording(rx, CFG, layout), layout


def test_sounding_symbol_occupies_only_its_bins():
    spec = np.abs(np.fft.fft(sounding_symbol(CFG)))
    bins = CFG.k0 + np.arange(CFG.n_sub)
    mask = np.zeros(CFG.period, bool)
    mask[bins] = mask[CFG.period - bins] = True
    assert spec[~mask].max() < 1e-9 * spec[mask].min()
    assert crest_factor_db(sounding_symbol(CFG)) < 6.0


def test_sync_recovers_latency():
    expected = (LATENCY + D_SM / SPEED_OF_SOUND) * CFG.fs
    tx, layout = build_tx(CFG, 3)
    rx = propagate(tx, CFG.fs, clutter_paths()[:1], noise_std=NOISE)
    assert abs(process_recording(rx, CFG, layout)["latency_samples"] - expected) <= 1
    # A strong echo one preamble symbol after the direct path biases the correlation
    # peak by a few samples; the impulse response is re-aligned on the direct path anyway.
    out, _ = run(3)
    assert abs(out["latency_samples"] - expected) <= CFG.period / CFG.n_sub / 3
    assert out["h"].shape[1] == CFG.n_sub


@pytest.mark.parametrize("r", [0.3, 0.6, 1.0])
def test_static_reflector_range_from_difference(r):
    empty, _ = run(4, seed=1)
    book, _ = run(4, [(0.05, reflector_delay(r, LATENCY))], seed=2)
    diff = np.abs(book["h"].mean(axis=0) - empty["h"].mean(axis=0))
    ranges = range_axis(CFG)
    diff[ranges < 0.15] = 0
    assert abs(ranges[np.argmax(diff)] - expected_range(r)) <= CFG.range_bin


def test_moving_reflector_track():
    def r_of_t(t):   # 0.3 m -> 0.7 m and back over 8 s, starting after the preamble
        u = np.clip((t - 1.2) / 8.0, 0, 1)
        return 0.3 + 0.4 * (1 - np.abs(1 - 2 * u))
    out, layout = run(10, [(0.05, reflector_delay(r_of_t, LATENCY))])
    h_dyn = remove_static(out["h"], CFG.frame_rate)
    ranges = range_axis(CFG)
    sel = ranges >= 0.15
    track = ranges[sel][np.argmax(np.abs(h_dyn[:, sel]), axis=1)]
    skip = int(np.ceil(0.3 * CFG.frame_rate))
    t = (out["start"] + (skip + np.arange(len(track)) + 0.5) * CFG.period) / CFG.fs
    truth = expected_range(r_of_t(t - LATENCY))
    moving = (t > 1.5) & (t < 9.0)
    err = np.abs(track[moving] - truth[moving])
    assert np.mean(err <= 1.5 * CFG.range_bin) > 0.9


@pytest.mark.parametrize("bpm", [10.0, 15.0, 20.0])
def test_breathing_rate(bpm):
    chest = breathing_range(0.4, 0.005, bpm)
    out, _ = run(30, [(0.03, reflector_delay(chest, LATENCY))], cue_bpm=bpm)
    est = breathing_rate(out["h"], CFG)
    assert abs(est["rate_bpm"] - bpm) < 0.5
    assert abs(est["range_m"] - expected_range(0.4)) <= 1.5 * CFG.range_bin


def test_deep_breathing_is_not_doubled():
    """A 2.6 cm peak-to-peak chest swing (> lambda/2) wraps the I/Q points around the
    circle; the rate must come from the arc phase, not a folded linear projection."""
    chest = breathing_range(0.4, 0.013, 10.0)
    out, _ = run(30, [(0.03, reflector_delay(chest, LATENCY))], cue_bpm=10.0)
    est = breathing_rate(out["h"], CFG)
    assert est["method"] == "arc"
    assert abs(est["rate_bpm"] - 10.0) < 0.5


def test_config_numbers():
    assert CFG.period == 768 and CFG.k0 == 96
    assert abs(CFG.frame_rate - 62.5) < 1e-9
    assert abs(CFG.range_bin - 0.0429) < 1e-3
    assert abs(CFG.wavelength - 0.0429) < 1e-3
    assert GUARD_TAPS < CFG.n_sub
