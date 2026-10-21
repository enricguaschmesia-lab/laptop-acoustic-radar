"""Quick quality check of one capture, run right after recording so a bad take can be redone."""

import numpy as np

from .config import SounderConfig
from .processing import breathing_rate, process_recording, range_axis, remove_static

MOTION_RANGE = (0.15, 2.5)      # m, where people can be in the protocol
MIN_SYNC_RATIO = 1.5            # correlation peak over the detection threshold
MAX_RATE_ERR = 1.5              # breaths/min, paced breathing vs the cue


def motion_level_db(h: np.ndarray, cfg: SounderConfig, window_s: float = 2.0) -> float:
    """Power of the clutter-removed response in the motion range, dB re the total received
    power (which, unlike the single direct-path tap, does not depend on sub-tap timing)."""
    h_dyn = remove_static(h, cfg.frame_rate, window_s=window_s)
    r = range_axis(cfg)
    sel = (r >= MOTION_RANGE[0]) & (r <= MOTION_RANGE[1])
    total = np.mean(np.sum(np.abs(h) ** 2, axis=1))
    return float(10 * np.log10(np.sum(np.mean(np.abs(h_dyn[:, sel]) ** 2, axis=0)) / total))


def quick_check(rec: np.ndarray, meta: dict) -> dict:
    cfg = SounderConfig.from_dict(meta["cfg"])
    layout = meta["layout"]
    dev = meta.get("device", {})
    res = {"peak": float(np.max(np.abs(rec))), "clipped": int(np.sum(np.abs(rec) >= 0.999)),
           "flags": [k for k in ("input_overflow", "input_underflow", "output_underflow", "output_overflow")
                     if dev.get(k)],
           "channels": []}
    cue = layout.get("cue_bpm")
    for ch in range(rec.shape[1]):
        c = {}
        try:
            p = process_recording(rec[:, ch], cfg, layout)
        except ValueError as e:
            c["error"] = str(e)
            res["channels"].append(c)
            continue
        c["sync_ratio"] = p["sync"]["peak_val"] / p["sync"]["threshold"]
        c["latency_ms"] = 1e3 * p["latency_samples"] / cfg.fs
        c["motion_db"] = motion_level_db(p["h"], cfg)
        if cue:
            est = breathing_rate(p["h"], cfg, min_range=0.15, max_range=1.6)
            c.update(rate_bpm=est["rate_bpm"], rate_err=est["rate_bpm"] - cue,
                     breath_range_m=est["range_m"], band_fraction=est["band_fraction"])
        res["channels"].append(c)

    warnings = []
    if res["peak"] < 0.02:
        warnings.append("very low level: are the speakers muted?")
    if res["clipped"]:
        warnings.append(f"{res['clipped']} clipped samples: lower the volume")
    if res["flags"]:
        warnings.append(f"dropped audio buffers ({', '.join(res['flags'])})")
    ok = [c for c in res["channels"] if "error" not in c]
    if len(ok) < len(res["channels"]):
        warnings.append("processing failed on a channel")
    if ok and max(c["sync_ratio"] for c in ok) < MIN_SYNC_RATIO:
        warnings.append("weak preamble sync")
    if cue and ok and min(abs(c["rate_err"]) for c in ok) > MAX_RATE_ERR:
        warnings.append(f"breathing rate off the {cue:.0f}/min cue on both channels")
    res["warnings"] = warnings
    return res


def format_check(qc: dict) -> str:
    parts = [f"peak {qc['peak']:.2f}"]
    for i, c in enumerate(qc["channels"]):
        if "error" in c:
            parts.append(f"ch{i} error")
            continue
        s = f"ch{i} motion {c['motion_db']:.1f} dB"
        if "rate_bpm" in c:
            s += f", {c['rate_bpm']:.1f}/min at {c['breath_range_m'] * 100:.0f} cm"
        parts.append(s)
    line = " | ".join(parts)
    return line + ("".join(f"\n  WARNING: {w}" for w in qc["warnings"]) if qc["warnings"] else "")
