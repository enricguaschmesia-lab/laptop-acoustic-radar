"""Static hardware check: play the sounder on the laptop speaker, record the laptop mic,
and report level, SNR, direct-path stability and the static clutter floor.

    python scripts/passive_check.py --seconds 15
"""

import argparse
import json
from pathlib import Path

from radar.audio_io import DEFAULT_HOSTAPI, playrec, save_capture
from radar.config import SounderConfig
from radar.diagnostics import analyze_static, plot_static, summary
from radar.waveform import build_tx

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--amplitude", type=float, default=0.5)
    ap.add_argument("--hostapi", default=DEFAULT_HOSTAPI, help="PortAudio host API (default: WASAPI on Windows)")
    ap.add_argument("--exclusive", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = SounderConfig(amplitude=args.amplitude)
    tx, layout = build_tx(cfg, args.seconds)
    print(f"Playing {len(tx) / cfg.fs:.1f} s on {args.hostapi or 'the default devices'}"
          f"{' (exclusive)' if args.exclusive else ''} ...")
    rec, dev = playrec(tx, cfg.fs, hostapi=args.hostapi, exclusive=args.exclusive)
    label = "passive" + (f"_{args.tag}" if args.tag else "")
    meta = {"label": label, "cfg": cfg.to_dict(), "layout": layout, "device": dev}
    stem = save_capture(ROOT / "data" / "raw" / "smoke", label, rec, cfg.fs, meta)

    out_dir = ROOT / "results" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"capture": str(stem), "device": dev, "channels": []}
    for ch in range(rec.shape[1]):
        res = analyze_static(rec[:, ch], layout, cfg)
        plot_static(res, cfg, f"{stem.name} - mic channel {ch}", out_dir / f"{stem.name}_ch{ch}.png")
        s = summary(res)
        report["channels"].append(s)
        print(f"\n--- mic channel {ch} ---")
        for k, v in s.items():
            if k != "checks":
                print(f"  {k:26s} {v:.4g}" if isinstance(v, float) else f"  {k:26s} {v}")
        for name, ok in s["checks"].items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    (out_dir / f"{stem.name}.json").write_text(json.dumps(report, indent=2, default=bool))
    print(f"\nSaved {stem}.wav and figures in {out_dir}")


if __name__ == "__main__":
    main()
