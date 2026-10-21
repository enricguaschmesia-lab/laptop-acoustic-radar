"""Hardware go/no-go for the laptop acoustic radar.

Record (plays sound; nothing plays until you press Enter at each step):
    python scripts/go_no_go.py
Re-analyse a saved session offline (no sound):
    python scripts/go_no_go.py --analyze data/raw/gonogo/<session>
Dry run on simulated captures (no sound):
    python scripts/go_no_go.py --simulate

Steps: empty room, book at 30 cm, book at 60 cm, book pulled on a string from 30 to 70 cm,
paced breathing at ~40 cm. GO if, on at least one mic channel, both books are detected
with the right range difference, the pull is tracked, and the breathing rate matches the cue.

Why the setup rules in PREFLIGHT matter: the sounder measures range only (everything at
the same distance in any direction shares a 4.3 cm bin), range wraps around every 2.74 m
(the sounding repeats every 16 ms), and the book test subtracts the empty-room capture,
so anything other than the book that moves or changes between captures looks like a target.
"""

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from radar.audio_io import countdown, load_capture, save_capture
from radar.config import SPEED_OF_SOUND, SounderConfig
from radar.diagnostics import analyze_static, summary
from radar.processing import breathing_rate, process_recording, range_axis, remove_static, static_difference
from radar.waveform import build_tx

ROOT = Path(__file__).resolve().parents[1]

CUE_BPM = 15.0
STEPS = [
    {"label": "empty", "seconds": 15, "countdown": 15, "away": True,
     "text": "Nothing but the table and the laptop within ~1 m of it; keep the book with you. During "
             "the countdown go to your waiting spot (outside the room is best) and stay still until "
             "the sound has stopped."},
    {"label": "book30", "seconds": 8, "countdown": 15, "away": True,
     "text": "Stand the book upright on the 30 cm mark, centred, face square to the laptop. Move nothing "
             "else. Then go back to the same waiting spot during the countdown."},
    {"label": "book60", "seconds": 8, "countdown": 15, "away": True,
     "text": "Move only the book to the 60 cm mark, same orientation. Then go back to the same waiting "
             "spot during the countdown."},
    {"label": "sweep", "seconds": 12, "countdown": 8,
     "text": "Stand the book on a placemat or piece of cardboard at the 30 cm mark (a small box behind it "
             "keeps it upright) and tape a ~1.5 m string to the far edge of the mat. During the countdown "
             "go to the far end of the table, in line with the laptop and 1.5-2 m from it. When the sound "
             "starts, pull the mat slowly and steadily to the 70 cm mark (about 8 s), then stay still. "
             "Keep your body still; only your hands move."},
    {"label": "breathing", "seconds": 40, "countdown": 8, "cue": CUE_BPM,
     "text": "Remove the book and mat. Place a chair so that, seated, your chest is ~40 cm from the "
             "screen, and tape its position for later sessions. Sit with your hands in your lap, head "
             "still, not talking. Breathe with the tone: inhale while the pitch rises, exhale while "
             "it falls (15 breaths/min)."},
]

PREFLIGHT = """\
Room setup (the sounder measures range only, in rings around the laptop, and range
wraps around every 2.74 m):
  - A normal furnished room (sofa, bed, curtains, rug). Clap once: if it rings for more
    than ~0.5 s, use another room or add blankets. Not a bathroom or empty room, not outside.
  - Door and windows closed; fans, air conditioning and fan heaters off; nobody else in
    the room, pets out; no music or TV; phone on silent.
  - Laptop near one end of a table out in the room (not a desk against a wall), facing
    along it, with >= 80 cm of free table in front. Lid at ~110 deg.
  - Nothing within ~1 m of the laptop in ANY direction except the table: move chairs, cups,
    bags and monitors away. Walls and wardrobes ideally >= 1-1.5 m away.
  - Tape marks at 30, 60 and 70 cm from the laptop's front edge, centred on the laptop.
  - Book: large, rigid and flat (hardcover A4 or bigger), upright, face square to the
    laptop (a few degrees of tilt can send the echo past the mic).
  - Pick ONE waiting spot for the empty and book steps: outside the room (best), or a taped
    spot >= 2 m away. Never 3-3.5 m in front of or behind the laptop: range wraps around
    there and puts you on top of the books.
  - Speakers unmuted, volume at the level noted last time, fixed for every session.
  - Between steps move only what the step says; never touch the laptop during a capture.
Each step plays sound from the laptop speakers, and only after you press Enter."""

# Decision thresholds
BOOK_DELTA_M, BOOK_TOL_M, MIN_PROMINENCE_DB = 0.30, 0.065, 6.0
SWEEP_SPAN_M = (0.25, 0.60)
BREATH_TOL_BPM = 1.0


# ----------------------------------------------------------------------------- record
def record(session_dir: Path, cfg: SounderConfig) -> None:
    from radar.audio_io import playrec
    print(PREFLIGHT)
    if input("\nPress Enter when the room is set up, 'q' to quit: ").strip().lower() == "q":
        return
    for i, step in enumerate(STEPS, 1):
        while True:
            print(f"\nStep {i}/{len(STEPS)}: {step['label']} ({step['seconds']} s of sound)")
            print(textwrap.fill(step["text"], 90, initial_indent="  ", subsequent_indent="  "))
            if step.get("away"):
                capture_s = cfg.lead_s + cfg.sc_nsyms / cfg.sc_f_sym + step["seconds"] + cfg.tail_s
                away = int(np.ceil(step["countdown"] + capture_s)) + 5
                print(f"  You get {step['countdown']} s to reach your waiting spot; stay there about "
                      f"{away} s in total (until the sound has stopped).")
            ans = input("  Press Enter to start, 's' to skip, 'q' to quit: ").strip().lower()
            if ans == "q":
                return
            if ans == "s":
                break
            countdown(step["countdown"])
            tx, layout = build_tx(cfg, step["seconds"], cue_bpm=step.get("cue"))
            rec, dev = playrec(tx, cfg.fs)
            meta = {"label": step["label"], "cfg": cfg.to_dict(), "layout": layout, "device": dev,
                    "instructions": step["text"]}
            stem = save_capture(session_dir, step["label"], rec, cfg.fs, meta)
            peak = float(np.abs(rec).max())
            flags = [k for k in ("input_overflow", "output_underflow") if dev.get(k)]
            print(f"  saved {stem.name}  peak {peak:.3f}" + (f"  WARNING {flags}" if flags else ""))
            if peak < 0.02:
                print("  WARNING: very low level - are the speakers muted?")
            if input("  Enter to continue, 'r' to redo this step: ").strip().lower() != "r":
                break


# ---------------------------------------------------------------------------- analyse
def latest(session_dir: Path, label: str) -> Path | None:
    stems = sorted(session_dir.glob(f"*_{label}.json"))
    return stems[-1].with_suffix("") if stems else None


def analyze_channel(session_dir: Path, ch: int) -> dict:
    res, out = {"channel": ch, "checks": {}}, {}
    for step in STEPS:
        stem = latest(session_dir, step["label"])
        if stem is None:
            continue
        rec, meta = load_capture(stem)
        cfg = SounderConfig.from_dict(meta["cfg"])
        out[step["label"]] = (process_recording(rec[:, ch], cfg, meta["layout"]), rec[:, ch], meta)
    if not out:
        raise FileNotFoundError(f"no captures in {session_dir}")
    r = range_axis(cfg)

    if "empty" in out:
        _, x, meta = out["empty"]
        res["empty"] = summary(analyze_static(x, meta["layout"], cfg))

    # Static books: difference against the empty room
    if "empty" in out:
        for lab in ("book30", "book60"):
            if lab in out:
                diff = static_difference(out["empty"][0]["H"], out[lab][0]["H"], cfg)
                sel = (r >= 0.15) & (r <= 1.5)
                k = np.flatnonzero(sel)[np.argmax(diff[sel])]
                res[lab] = {"range_m": float(r[k]),
                            "prominence_db": float(20 * np.log10(diff[k] / np.median(diff[sel]))),
                            "profile": diff.tolist()}
        if "book30" in res and "book60" in res:
            delta = res["book60"]["range_m"] - res["book30"]["range_m"]
            res["book_delta_m"] = delta
            res["checks"]["books detected (>= 6 dB)"] = min(res["book30"]["prominence_db"],
                                                            res["book60"]["prominence_db"]) >= MIN_PROMINENCE_DB
            res["checks"][f"book range difference {BOOK_DELTA_M:.2f} +/- {BOOK_TOL_M:.3f} m"] = \
                abs(delta - BOOK_DELTA_M) <= BOOK_TOL_M

    # Pulled book: ridge of the clutter-removed range-time map. The operator stands
    # 1.5-2 m away, beyond the 1.2 m search window.
    if "sweep" in out:
        h_dyn = remove_static(out["sweep"][0]["h"], cfg.frame_rate)
        sel = (r >= 0.1) & (r <= 1.2)
        mag = np.abs(h_dyn[:, sel])
        track = r[sel][np.argmax(mag, axis=1)]
        strong = mag.max(axis=1) >= 2 * np.median(mag, axis=1)       # peak >= 6 dB above the floor
        track_s = median_filter(track, 9)
        valid = track_s[strong] if strong.sum() > 20 else track_s
        span = float(np.percentile(valid, 95) - np.percentile(valid, 5))
        res["sweep"] = {"span_m": span, "p5_m": float(np.percentile(valid, 5)),
                        "p95_m": float(np.percentile(valid, 95)), "track": track_s.tolist(),
                        "strong_fraction": float(strong.mean())}
        res["checks"][f"pull tracked (span {SWEEP_SPAN_M[0]}-{SWEEP_SPAN_M[1]} m)"] = \
            SWEEP_SPAN_M[0] <= span <= SWEEP_SPAN_M[1]

    # Paced breathing
    if "breathing" in out:
        meta = out["breathing"][2]
        cue = meta["layout"].get("cue_bpm") or CUE_BPM
        est = breathing_rate(out["breathing"][0]["h"], cfg, min_range=0.15, max_range=1.2)
        res["breathing"] = {"rate_bpm": est["rate_bpm"], "cue_bpm": cue, "range_m": est["range_m"],
                            "band_fraction": est["band_fraction"], "signal": est["signal"].tolist()}
        res["checks"][f"breathing within {BREATH_TOL_BPM:.0f} bpm of the cue"] = \
            abs(est["rate_bpm"] - cue) <= BREATH_TOL_BPM
    res["go"] = bool(res["checks"]) and all(res["checks"].values())
    return res


def plot_channel(res: dict, cfg: SounderConfig, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = range_axis(cfg)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for lab, c in (("book30", "C0"), ("book60", "C1")):
        if lab in res:
            p = np.asarray(res[lab]["profile"])
            ax[0, 0].plot(r * 100, 20 * np.log10(p / p.max()), color=c,
                          label=f"{lab}: {res[lab]['range_m'] * 100:.0f} cm")
            ax[0, 0].axvline(res[lab]["range_m"] * 100, color=c, ls="--", lw=0.8)
    ax[0, 0].set(xlabel="Range (cm)", ylabel="dB", xlim=(0, 150), title="Book vs empty room (static difference)")
    ax[0, 0].legend()

    if "sweep" in res:
        t = np.arange(len(res["sweep"]["track"])) / cfg.frame_rate
        ax[0, 1].plot(t, np.asarray(res["sweep"]["track"]) * 100)
        ax[0, 1].set(xlabel="Time (s)", ylabel="Range (cm)",
                     title=f"Pulled book, tracked span {res['sweep']['span_m'] * 100:.0f} cm")
    if "breathing" in res:
        s = np.asarray(res["breathing"]["signal"])
        t = np.arange(len(s)) / cfg.frame_rate
        ax[1, 0].plot(t, s)
        ax[1, 0].set(xlabel="Time (s)", ylabel="Projected I/Q (a.u.)",
                     title=f"Chest bin at {res['breathing']['range_m'] * 100:.0f} cm")
        f = np.fft.rfftfreq(1 << 16, 1 / cfg.frame_rate)
        spec = np.abs(np.fft.rfft((s - s.mean()) * np.hanning(len(s)), 1 << 16)) ** 2
        band = f <= 1.0
        ax[1, 1].plot(f[band] * 60, spec[band] / spec[band].max())
        ax[1, 1].axvline(res["breathing"]["cue_bpm"], color="k", ls="--", label="cue")
        ax[1, 1].set(xlabel="Breaths per minute", ylabel="Power (norm.)",
                     title=f"Estimated {res['breathing']['rate_bpm']:.1f} bpm")
        ax[1, 1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def analyze(session_dir: Path) -> bool:
    out_dir = ROOT / "results" / "gonogo"
    out_dir.mkdir(parents=True, exist_ok=True)
    report, any_go = {"session": str(session_dir), "channels": []}, False
    stem = latest(session_dir, "empty") or next(session_dir.glob("*.json")).with_suffix("")
    cfg = SounderConfig.from_dict(load_capture(stem)[1]["cfg"])
    n_ch = load_capture(stem)[0].shape[1]
    for ch in range(n_ch):
        res = analyze_channel(session_dir, ch)
        plot_channel(res, cfg, out_dir / f"{session_dir.name}_ch{ch}.png", f"{session_dir.name} - mic channel {ch}")
        print(f"\n=== mic channel {ch} ===")
        if "empty" in res:
            e = res["empty"]
            print(f"  empty room: SNR {e['inband_snr_db']:.1f} dB, amp std {e['amp_rel_std']:.1%}, "
                  f"phase jitter {e['phase_std_deg']:.1f} deg, floor {e['floor_db']:.1f} dB")
        for lab in ("book30", "book60"):
            if lab in res:
                print(f"  {lab}: {res[lab]['range_m'] * 100:.0f} cm, prominence {res[lab]['prominence_db']:.1f} dB")
        if "book_delta_m" in res:
            print(f"  book60 - book30 = {res['book_delta_m'] * 100:.1f} cm (expected 30)")
        if "sweep" in res:
            print(f"  pull: {res['sweep']['p5_m'] * 100:.0f} -> {res['sweep']['p95_m'] * 100:.0f} cm")
        if "breathing" in res:
            b = res["breathing"]
            print(f"  breathing: {b['rate_bpm']:.2f} bpm (cue {b['cue_bpm']:.0f}) at {b['range_m'] * 100:.0f} cm")
        for name, ok in res["checks"].items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"  -> {'GO' if res['go'] else 'NO-GO'}")
        any_go |= res["go"]
        # the JSON report keeps the scalars; long traces stay in the figures
        steps = ("sweep", "breathing", "book30", "book60")
        traces = ("profile", "track", "signal")
        report["channels"].append({k: v for k, v in res.items() if k not in steps}
                                  | {k: {kk: vv for kk, vv in res[k].items() if kk not in traces}
                                     for k in steps if k in res})
    report["go"] = any_go
    (out_dir / f"{session_dir.name}.json").write_text(json.dumps(report, indent=2, default=bool))
    print(f"\nOVERALL: {'GO' if any_go else 'NO-GO'}  (figures in {out_dir})")
    return any_go


# --------------------------------------------------------------------------- simulate
def simulate(session_dir: Path, cfg: SounderConfig, seed: int = 0) -> None:
    """Write synthetic captures for every step (two mic channels), no audio involved."""
    from radar.scenes import LATENCY as latency
    from radar.scenes import MIC_OFFSETS
    from radar.sim import breathing_range, propagate, reflector_delay

    def scene(label, ch, t0):
        d = MIC_OFFSETS[ch]
        paths = [(1.0, latency + d / SPEED_OF_SOUND), (0.3, reflector_delay(0.12, latency)),
                 (0.1, reflector_delay(1.9, latency)), (0.05, reflector_delay(2.4, latency))]
        if label == "book30":
            paths.append((0.03, reflector_delay(0.30, latency)))
        elif label == "book60":
            paths.append((0.03, reflector_delay(0.60, latency)))
        elif label == "sweep":
            def r_t(t):   # pulled one way, 30 -> 70 cm over 8 s, then at rest
                return 0.30 + 0.40 * np.clip((t - t0) / 8.0, 0, 1)
            paths.append((0.03, reflector_delay(r_t, latency)))
            # operator at the far end of the table, breathing
            paths.append((0.01, reflector_delay(breathing_range(1.75, 0.005, 14.0), latency)))
        elif label == "breathing":
            paths.append((0.02, reflector_delay(breathing_range(0.40, 0.005, CUE_BPM), latency)))
        return paths

    for i, step in enumerate(STEPS):
        tx, layout = build_tx(cfg, step["seconds"], cue_bpm=step.get("cue"))
        t0 = (layout["sound_start"] / cfg.fs) + latency + 1.0
        rec = np.column_stack([propagate(tx, cfg.fs, scene(step["label"], ch, t0),
                                         noise_std=2e-3, seed=seed + 10 * i + ch) for ch in (0, 1)])
        meta = {"label": step["label"], "cfg": cfg.to_dict(), "layout": layout,
                "device": {"simulated": True}, "instructions": step["text"]}
        save_capture(session_dir, step["label"], rec, cfg.fs, meta)
        time.sleep(1.1)   # distinct timestamps in the file names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyze", type=Path, help="analyse a saved session directory (no sound)")
    ap.add_argument("--simulate", action="store_true", help="dry run on simulated captures (no sound)")
    ap.add_argument("--amplitude", type=float, default=0.5)
    args = ap.parse_args()
    cfg = SounderConfig(amplitude=args.amplitude)

    if args.analyze:
        sys.exit(0 if analyze(args.analyze) else 1)
    session = ROOT / "data" / "raw" / "gonogo" / (("sim-" if args.simulate else "") + time.strftime("%Y%m%d-%H%M%S"))
    if args.simulate:
        simulate(session, cfg)
    else:
        record(session, cfg)
    analyze(session)


if __name__ == "__main__":
    main()
