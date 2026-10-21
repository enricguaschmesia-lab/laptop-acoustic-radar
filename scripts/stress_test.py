"""Stress tests: macro-F1 as noise is added to the measured responses, for 4 s and 1 s windows.

Complex white Gaussian noise is added to the aligned impulse responses of every capture before
windowing, at a level in dB relative to that capture's total received power: the same reference
as the motion levels (empty room ~ -46 dB, still ~ -23, waving ~ -20, walking ~ -18 dB). Train
and test sessions get noise at the same level (matched conditions); evaluation is
leave-one-session-out as in train_eval.py, with one CNN seed per condition.

    python scripts/stress_test.py
    python scripts/stress_test.py --quick      # 1 epoch, 2 levels, separate output
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from radar.audio_io import load_capture
from radar.config import SounderConfig
from radar.evaluation import loso_classical, loso_cnn
from radar.features import FEATURE_NAMES, window_features, window_maps, window_starts
from radar.processing import process_recording
from radar.protocol import CLASSES, N_SESSIONS

ROOT = Path(__file__).resolve().parents[1]

CACHE = ROOT / "data" / "processed" / "h_cache.npz"
WINDOWS = {"4s": (256, 64), "1s": (64, 32)}
LEVELS = [None, -30, -20, -15, -10, -5, 0]
# Dataset-mean motion levels of the occupied classes (the empty room sits at -45.7 dB, off the action)
CLASS_LEVELS = {"still": -22.7, "waving": -20.1, "walking": -17.8}
MODEL_LABELS = {"energy_lr": "Energy only (LR)", "features_lr": "Features + LR",
                "features_rf": "Features + RF", "cnn": "Small CNN"}


def load_responses():
    """Aligned impulse responses of every capture in use, (n_mics, T, n_sub) each, cached."""
    if CACHE.exists():
        d = np.load(CACHE)
        cfg = SounderConfig.from_dict(json.loads(str(d["cfg"])))
        n = int(d["n"])
        return cfg, [(d[f"h{i}"], int(d["cls"][i]), int(d["session"][i]), str(d["capture"][i])) for i in range(n)]
    out, cfg = [], None
    for s in range(1, N_SESSIONS + 1):
        sdir = ROOT / "data" / "raw" / "dataset" / f"session_{s}"
        for c in json.loads((sdir / "manifest.json").read_text())["captures"]:
            if c["superseded"]:
                continue
            rec, meta = load_capture(sdir / c["stem"])
            cfg = SounderConfig.from_dict(meta["cfg"])
            hs = [process_recording(rec[:, ch], cfg, meta["layout"])["h"] for ch in range(rec.shape[1])]
            n = min(len(h) for h in hs)
            out.append((np.stack([h[:n] for h in hs]).astype(np.complex64), CLASSES.index(c["cls"]), s,
                        f"s{s}_{c['label']}"))
            print(f"  cached s{s} {c['label']}")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, n=len(out), cfg=json.dumps(cfg.to_dict()), cls=[o[1] for o in out],
             session=[o[2] for o in out], capture=[o[3] for o in out], **{f"h{i}": o[0] for i, o in enumerate(out)})
    return cfg, out


def build_windows(responses, cfg, level_db, win, hop, seed=1234):
    maps, feats, y, sess = [], [], [], []
    for i, (hs, cls, s, _) in enumerate(responses):
        hs = hs.astype(np.complex128)
        if level_db is not None:
            rng = np.random.default_rng(seed + i)
            p_total = np.mean(np.sum(np.abs(hs) ** 2, axis=2), axis=1)          # per mic
            sigma = np.sqrt(10 ** (level_db / 10) * p_total / hs.shape[2])
            hs = hs + sigma[:, None, None] * (rng.standard_normal(hs.shape)
                                              + 1j * rng.standard_normal(hs.shape)) / np.sqrt(2)
        for st in window_starts(hs.shape[1], win, hop):
            hw = [h[st: st + win] for h in hs]
            maps.append(window_maps(hw))
            feats.append(window_features(hw, cfg))
            y.append(cls)
            sess.append(s)
    return np.asarray(maps, np.float32), np.asarray(feats, np.float32), np.asarray(y), np.asarray(sess)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--quick", action="store_true", help="1 epoch, two levels, 1 s windows only")
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from saved results")
    args = ap.parse_args()
    out_dir = ROOT / "results" / "stress"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("quick.json" if args.quick else "results.json")
    results = json.loads(path.read_text()) if path.exists() else {}
    if args.plot_only:
        plot(results, out_dir / ("quick.png" if args.quick else "stress.png"))
        return
    windows = {"1s": WINDOWS["1s"]} if args.quick else WINDOWS
    levels = [None, -20] if args.quick else LEVELS
    epochs = 1 if args.quick else args.epochs

    cfg, responses = load_responses()
    i_energy, empty, n_cls = FEATURE_NAMES.index("motion_db"), CLASSES.index("empty"), len(CLASSES)
    for wname, (win, hop) in windows.items():
        for level in levels:
            key = f"{wname}|{'none' if level is None else level}"
            if key in results:
                continue
            t0 = time.time()
            maps, feats, y, sess = build_windows(responses, cfg, level, win, hop)
            r = loso_classical(feats, y, sess, i_energy, empty, n_cls)
            r.update(loso_cnn(maps, y, sess, empty, n_cls, seed=0, epochs=epochs))
            results[key] = {"window": wname, "level_db": level, "n_windows": int(len(y)),
                            **{m: {str(k): v for k, v in folds.items()} for m, folds in r.items()}}
            path.write_text(json.dumps(results, indent=2))
            mean = {m: np.mean(list(results[key][m].values())) for m in r}
            print(f"{key:9s} {len(y):5d} windows {time.time() - t0:5.0f} s | macro-F1 "
                  + ", ".join(f"{m} {mean[m]:.3f}" for m in ("energy_lr", "features_lr", "features_rf", "cnn"))
                  + f" | presence energy {mean['presence_energy']:.3f}, CNN {mean['presence_cnn']:.3f}", flush=True)
    plot(results, out_dir / ("quick.png" if args.quick else "stress.png"))


def plot(results: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wins = [w for w in WINDOWS if any(r["window"] == w for r in results.values())]
    fig, ax = plt.subplots(2, len(wins), figsize=(7 * len(wins), 8.5), squeeze=False)
    none_x = -50
    for j, w in enumerate(wins):
        rows = sorted((r for r in results.values() if r["window"] == w),
                      key=lambda r: none_x if r["level_db"] is None else r["level_db"])
        x = [none_x if r["level_db"] is None else r["level_db"] for r in rows]
        for row, models in ((0, ("energy_lr", "features_lr", "features_rf", "cnn")),
                            (1, ("presence_energy", "presence_cnn"))):
            a = ax[row, j]
            for m in models:
                vals = np.array([list(r[m].values()) for r in rows])
                label = MODEL_LABELS.get(m, {"presence_energy": "Energy threshold",
                                             "presence_cnn": "Small CNN"}.get(m, m))
                a.plot(x, vals.mean(axis=1), marker="o", label=label)
                a.fill_between(x, vals.min(axis=1), vals.max(axis=1), alpha=0.15)
            for cls, lvl in CLASS_LEVELS.items():
                a.axvline(lvl, color="grey", ls=":", lw=0.8)
                a.text(lvl + 0.4, 0.03, f"{cls} level", rotation=90, fontsize=7, ha="left", va="bottom",
                       color="grey", transform=a.get_xaxis_transform())
            a.set(xlabel="added noise (dB re total received power)", ylim=(0, 1.1),
                  ylabel="macro-F1 (4 classes)" if row == 0 else "presence balanced accuracy",
                  title=f"{'Activity' if row == 0 else 'Presence'}, {w} windows")
            ticks = sorted(set(x))
            a.set_xticks(ticks, ["none" if t == none_x else str(t) for t in ticks])
            a.legend(fontsize=8, loc="lower left")
    fig.suptitle("Noise stress test, leave-one-session-out (band: min-max over test sessions)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
