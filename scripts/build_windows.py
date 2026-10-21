"""Cut every capture in use into 4.1 s windows (1 s hop) and cache maps, features and labels.

    python scripts/build_windows.py            # real dataset
    python scripts/build_windows.py --simulated
"""

import argparse
import json
from pathlib import Path

import numpy as np

from radar.audio_io import load_capture
from radar.config import SounderConfig
from radar.features import FEATURE_NAMES, HOP, WIN, window_features, window_maps, window_starts
from radar.processing import process_recording
from radar.protocol import CLASSES, N_SESSIONS

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulated", action="store_true")
    args = ap.parse_args()
    base = ROOT / "data" / "raw" / ("dataset_sim" if args.simulated else "dataset")
    out = ROOT / "data" / "processed" / ("windows_sim.npz" if args.simulated else "windows.npz")

    maps, feats, y, session, capture, start, distance, cue = [], [], [], [], [], [], [], []
    for s in range(1, N_SESSIONS + 1):
        sdir = base / f"session_{s}"
        manifest = json.loads((sdir / "manifest.json").read_text())
        for c in manifest["captures"]:
            if c["superseded"]:
                continue
            rec, meta = load_capture(sdir / c["stem"])
            cfg = SounderConfig.from_dict(meta["cfg"])
            hs = [process_recording(rec[:, ch], cfg, meta["layout"])["h"] for ch in range(rec.shape[1])]
            n = min(len(h) for h in hs)
            starts = list(window_starts(n))
            for st in starts:
                hw = [h[st: st + WIN] for h in hs]
                maps.append(window_maps(hw))
                feats.append(window_features(hw, cfg))
                y.append(CLASSES.index(c["cls"]))
                session.append(s)
                capture.append(f"s{s}_{c['label']}")
                start.append(st)
                distance.append(c["distance_m"] if c["distance_m"] is not None else np.nan)
                cue.append(c["cue_bpm"] if c["cue_bpm"] is not None else np.nan)
            print(f"  s{s} {c['label']:20s} {len(starts):3d} windows")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, maps=np.asarray(maps, np.float16), feats=np.asarray(feats, np.float32),
                        feature_names=np.asarray(FEATURE_NAMES), classes=np.asarray(CLASSES),
                        y=np.asarray(y), session=np.asarray(session), capture=np.asarray(capture),
                        start=np.asarray(start), distance=np.asarray(distance), cue=np.asarray(cue),
                        win=WIN, hop=HOP)
    y = np.asarray(y)
    print(f"\n{len(y)} windows -> {out}")
    for k, name in enumerate(CLASSES):
        print(f"  {name:8s} {np.sum(y == k):4d}")


if __name__ == "__main__":
    main()
