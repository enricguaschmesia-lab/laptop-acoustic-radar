"""Window maps and hand-crafted features on synthetic targets with known motion."""

import numpy as np

from radar.config import SPEED_OF_SOUND, SounderConfig
from radar.features import FEATURE_NAMES, WIN, doppler_axis, window_features, window_maps, window_starts
from radar.processing import GUARD_TAPS, process_recording
from radar.sim import propagate, reflector_delay
from radar.waveform import build_tx

CFG = SounderConfig()
LATENCY = 0.023


def _h(paths, seconds=6, seed=0):
    tx, layout = build_tx(CFG, seconds)
    rx = propagate(tx, CFG.fs, paths, noise_std=1e-3, seed=seed)
    return process_recording(rx, CFG, layout)["h"]


def _static():
    return [(1.0, LATENCY + 0.15 / SPEED_OF_SOUND), (0.3, reflector_delay(0.25, LATENCY))]


def test_window_starts():
    assert list(window_starts(WIN + 130, hop=64)) == [0, 64, 128]


def test_map_shapes():
    h = _h(_static())
    maps = window_maps([h[:WIN], h[:WIN]])
    assert maps.shape == (4, WIN // 2, CFG.n_sub - GUARD_TAPS)
    assert np.isfinite(maps).all()


def test_constant_speed_reflector_peaks_at_its_doppler():
    v = 0.3                                            # m/s, receding
    r = lambda t: 0.6 + v * (t - 1.0)
    h = _h(_static() + [(0.05, reflector_delay(r, LATENCY))])
    rd = window_maps([h[:WIN]])[0]                     # Doppler x range
    fd = doppler_axis(CFG)[::2] + (doppler_axis(CFG)[1] - doppler_axis(CFG)[0]) / 2
    peak_fd = fd[np.unravel_index(np.argmax(rd), rd.shape)[0]]
    expected = 2 * v * CFG.f_center / SPEED_OF_SOUND  # 14 Hz
    assert abs(abs(peak_fd) - expected) < 1.5


def test_features_separate_static_and_moving_scenes():
    static = _h(_static(), seed=1)
    walker = _h(_static() + [(0.05, reflector_delay(lambda t: 1.0 + 0.4 * np.sin(2 * np.pi * t / 4), LATENCY))], seed=2)
    f_static = dict(zip(FEATURE_NAMES, window_features([static[:WIN]], CFG), strict=True))
    f_walk = dict(zip(FEATURE_NAMES, window_features([walker[:WIN]], CFG), strict=True))
    assert f_walk["motion_db"] > f_static["motion_db"] + 20
    assert f_walk["mean_speed"] > 0.2
    # a sinusoidal swing dwells at its extremes (0.6 and 1.4 m, minus half the 0.15 m direct path)
    assert 0.45 < f_walk["range_peak_m"] < 1.4
    assert f_walk["range_spread_m"] > 0.15
