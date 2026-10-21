"""Protocol schedule, per-capture quality check and manifest bookkeeping."""

import importlib.util
from pathlib import Path

import numpy as np

from radar.config import SounderConfig
from radar.protocol import CLASSES, DISTANCES, N_SESSIONS, RATES, breathing_schedule, session_steps
from radar.qc import quick_check
from radar.scenes import scene_paths
from radar.sim import propagate
from radar.waveform import build_tx

CFG = SounderConfig()


def _load_collect():
    path = Path(__file__).resolve().parents[1] / "scripts" / "collect.py"
    spec = importlib.util.spec_from_file_location("collect", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sessions_1_to_3_cover_every_distance_rate_pair_once():
    pairs = [p for s in (1, 2, 3) for p in breathing_schedule(s)]
    assert sorted(pairs) == sorted((d, r) for d in DISTANCES for r in RATES)
    assert breathing_schedule(4) == breathing_schedule(1)


def test_steps_are_well_formed():
    for s in range(1, N_SESSIONS + 1):
        steps = session_steps(s)
        labels = [st["label"] for st in steps]
        assert len(labels) == len(set(labels))
        assert {st["cls"] for st in steps} == set(CLASSES)
        for st in steps:
            assert st["seconds"] > 0 and st["countdown"] > 0
            if st.get("cue"):
                assert st["distance_m"] in DISTANCES


def _simulated(step, seconds, session=1, seed=3):
    step = dict(step, seconds=seconds)
    tx, layout = build_tx(CFG, seconds, cue_bpm=step.get("cue"))
    rec = np.column_stack([propagate(tx, CFG.fs, scene_paths(step, ch, session, seed), noise_std=2e-3, seed=ch)
                           for ch in (0, 1)])
    return rec, {"cfg": CFG.to_dict(), "layout": layout, "device": {"simulated": True}}


def test_quick_check_on_paced_breathing():
    step = next(st for st in session_steps(2) if st.get("cue"))
    rec, meta = _simulated(step, 30)
    qc = quick_check(rec, meta)
    assert qc["warnings"] == []
    assert all(abs(c["rate_err"]) < 1.0 for c in qc["channels"])


def test_quick_check_flags_muted_speakers():
    tx, layout = build_tx(CFG, 3)
    rec = 1e-4 * np.random.default_rng(0).standard_normal((len(tx), 2))
    qc = quick_check(rec, {"cfg": CFG.to_dict(), "layout": layout, "device": {}})
    assert any("muted" in w for w in qc["warnings"])


def test_motion_level_orders_classes():
    steps = {st["cls"]: st for st in session_steps(1)}
    level = {cls: np.mean([c["motion_db"] for c in quick_check(*_simulated(steps[cls], 8))["channels"]])
             for cls in ("empty", "still", "waving")}
    assert level["empty"] + 6 < min(level["still"], level["waving"])


def test_redo_supersedes_previous_capture():
    collect = _load_collect()
    manifest = {"session": 1, "captures": []}
    step = session_steps(1)[0]
    collect.add_capture(manifest, step, Path("a"), {"warnings": []})
    collect.add_capture(manifest, step, Path("b"), {"warnings": []})
    assert collect.current(manifest, step["label"])["stem"] == "b"
    assert [c["superseded"] for c in manifest["captures"]] == [True, False]
