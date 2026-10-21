"""Simulated scenes for each protocol step, for dry runs and pipeline tests.

Each scene is a list of (gain, delay) paths for radar.sim.propagate. Moving parts are pure
functions of time so the chunked simulator can evaluate them piecewise. Gains follow
two-way spherical spreading (amplitude ~ 1/r^2) around rough values seen on hardware.
"""

import numpy as np

from .config import SPEED_OF_SOUND
from .sim import reflector_delay

LATENCY = 0.136                 # s, measured on the laptop
MIC_OFFSETS = (0.20, 0.22)      # speaker -> mic path of the two mic channels (m)
CHEST_GAIN, HEAD_GAIN, LIMB_GAIN, HAND_GAIN = 0.02, 0.008, 0.006, 0.01
BODY_GAIN = 0.06                # whole standing body, about three chest patches


def _spread(gain_at_ref: float, ref_m: float, r):
    return gain_at_ref * (ref_m / np.maximum(r, 0.1)) ** 2


def _sum_of_sines(rng, n: int, f_max: float, scale: float):
    """Smooth random displacement: a few random low-frequency sinusoids, RMS ~ scale."""
    f = rng.uniform(0.02, f_max, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    a = rng.normal(0, 1, n)
    norm = scale / np.sqrt(np.sum(a ** 2) / 2)
    return lambda t: norm * np.sum(a[:, None] * np.sin(2 * np.pi * f[:, None] * np.atleast_1d(t) + ph[:, None]), axis=0)


def static_paths(channel: int, session: int) -> list:
    """Direct path, table, and a few session-specific room reflectors (walls, furniture)."""
    rng = np.random.default_rng(1000 + session)
    paths = [(1.0, LATENCY + MIC_OFFSETS[channel] / SPEED_OF_SOUND),
             (0.3, reflector_delay(0.12, LATENCY))]
    for r, g in zip(rng.uniform(1.2, 2.6, 4), rng.uniform(0.02, 0.1, 4), strict=True):
        paths.append((float(g), reflector_delay(float(r), LATENCY)))
    return paths


def _seated_person(rng, distance: float, bpm: float | None, amp: float) -> list:
    """Chest and head of a seated person; bpm=None means natural, slightly irregular breathing."""
    if bpm is None:
        f0 = rng.uniform(12, 17) / 60
        beta, fm, phm = 0.8, rng.uniform(0.02, 0.05), rng.uniform(0, 2 * np.pi)
        phase = lambda t: 2 * np.pi * f0 * t + beta * np.sin(2 * np.pi * fm * t + phm)
    else:
        ph0 = rng.uniform(0, 2 * np.pi)
        phase = lambda t: 2 * np.pi * bpm / 60 * t + ph0
    sway = _sum_of_sines(rng, 5, 0.15, 0.002)
    chest = lambda t: distance + amp * np.sin(phase(t)) + sway(t)
    head = lambda t: distance + 0.06 + 0.2 * amp * np.sin(phase(t)) + sway(t)
    return [(_spread(CHEST_GAIN, 0.7, distance), reflector_delay(chest, LATENCY)),
            (_spread(HEAD_GAIN, 0.7, distance), reflector_delay(head, LATENCY))]


def _walker(rng, zone=(0.8, 2.5)) -> list:
    """Torso walking back and forth across the zone, plus two limbs swinging at the step rate.

    Path gains are constant in the simulator, so they use the range at the zone centre.
    """
    centre, half = (zone[0] + zone[1]) / 2, (zone[1] - zone[0]) / 2
    period = rng.uniform(6.0, 8.0)
    ph0, step_f = rng.uniform(0, 2 * np.pi), rng.uniform(0.8, 1.1)
    torso = lambda t: centre + half * np.sin(2 * np.pi * t / period + ph0)
    limb1 = lambda t: torso(t) + 0.15 * np.sin(2 * np.pi * step_f * t)
    limb2 = lambda t: torso(t) - 0.15 * np.sin(2 * np.pi * step_f * t)
    g_torso, g_limb = _spread(BODY_GAIN, 0.7, centre), _spread(LIMB_GAIN, 0.7, centre)
    return [(g_torso, reflector_delay(torso, LATENCY)),
            (g_limb, reflector_delay(limb1, LATENCY)),
            (g_limb, reflector_delay(limb2, LATENCY))]


def _hand(rng, base: float = 0.45, depth: float = 0.12) -> list:
    f, fm, ph = rng.uniform(1.0, 1.5), rng.uniform(0.05, 0.15), rng.uniform(0, 2 * np.pi)
    hand = lambda t: base + depth * np.sin(2 * np.pi * f * t + 2.0 * np.sin(2 * np.pi * fm * t) + ph)
    return [(_spread(HAND_GAIN, base, base), reflector_delay(hand, LATENCY))]


def scene_paths(step: dict, channel: int, session: int, seed: int) -> list:
    """All paths for one protocol step on one mic channel. The same seed gives the same
    person motion on both channels."""
    rng = np.random.default_rng(seed)
    paths = static_paths(channel, session)
    cls = step["cls"]
    if cls == "still":
        amp = 0.005 if step.get("cue") else 0.003
        paths += _seated_person(rng, step["distance_m"], step.get("cue"), amp)
    elif cls == "walking":
        paths += _walker(rng)
    elif cls == "waving":
        paths += _seated_person(rng, step["distance_m"], None, 0.003)
        paths += _hand(rng)
    return paths
