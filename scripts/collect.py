"""Labelled dataset collection for the laptop acoustic radar (protocol in radar/protocol.py).

Record a session (plays sound; nothing plays until you press Enter at each step):
    python scripts/collect.py --session 1
Redo a single step of a session (plays sound):
    python scripts/collect.py --session 2 --step breath_100cm_10bpm
Re-check a recorded session offline (no sound):
    python scripts/collect.py --session 1 --check
Dry run on simulated captures, written to data/raw/dataset_sim (no sound):
    python scripts/collect.py --session 1 --simulate

A session can be interrupted and restarted: steps already recorded are offered as
"keep or redo". Every capture is kept on disk; a redo only marks the old one superseded
in the session's manifest.json.
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

from radar.audio_io import countdown, load_capture, save_capture
from radar.config import SounderConfig
from radar.protocol import (
    N_SESSIONS,
    PREFLIGHT,
    PROTOCOL_VERSION,
    breathing_schedule,
    session_sound_seconds,
    session_steps,
)
from radar.qc import format_check, quick_check
from radar.waveform import build_tx

ROOT = Path(__file__).resolve().parents[1]

STEP_OVERHEAD_S = 25    # moving the chair, reading instructions, saving


# --------------------------------------------------------------------------- manifest
def session_dir(session: int, simulated: bool) -> Path:
    return ROOT / "data" / "raw" / ("dataset_sim" if simulated else "dataset") / f"session_{session}"


def load_manifest(sdir: Path, session: int, simulated: bool) -> dict:
    path = sdir / "manifest.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"session": session, "protocol_version": PROTOCOL_VERSION, "simulated": simulated, "captures": []}


def save_manifest(sdir: Path, manifest: dict) -> None:
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=bool))


def current(manifest: dict, label: str) -> dict | None:
    live = [c for c in manifest["captures"] if c["label"] == label and not c["superseded"]]
    return live[-1] if live else None


def add_capture(manifest: dict, step: dict, stem: Path, qc: dict) -> None:
    for c in manifest["captures"]:
        if c["label"] == step["label"]:
            c["superseded"] = True
    manifest["captures"].append({
        "label": step["label"], "cls": step["cls"], "distance_m": step.get("distance_m"),
        "cue_bpm": step.get("cue"), "seconds": step["seconds"], "stem": stem.name,
        "superseded": False, "qc": qc})


def capture_meta(step: dict, session: int, cfg: SounderConfig, layout: dict, device: dict) -> dict:
    return {"label": step["label"], "cls": step["cls"], "session": session,
            "protocol_version": PROTOCOL_VERSION, "distance_m": step.get("distance_m"),
            "cue_bpm": step.get("cue"), "cfg": cfg.to_dict(), "layout": layout, "device": device,
            "instructions": step["text"]}


# ----------------------------------------------------------------------------- record
def describe_session(session: int) -> str:
    steps = session_steps(session)
    sound = session_sound_seconds(session)
    total = sound + sum(s["countdown"] for s in steps) + STEP_OVERHEAD_S * len(steps)
    sched = ", ".join(f"{d * 100:.0f} cm at {b:.0f}/min" for d, b in breathing_schedule(session))
    return (f"Session {session}/{N_SESSIONS}: {len(steps)} steps, {sound / 60:.1f} min of sound, "
            f"about {total / 60:.0f} min in total.\n  Paced breathing: {sched}.")


def select_steps(session: int, only: str | None) -> list[dict]:
    steps = session_steps(session)
    if only is None:
        return steps
    chosen = [s for s in steps if s["label"] == only]
    if not chosen:
        sys.exit(f"no step {only!r} in session {session}; steps: {', '.join(s['label'] for s in steps)}")
    return chosen


def record(session: int, cfg: SounderConfig, only: str | None = None) -> None:
    from radar.audio_io import playrec
    sdir = session_dir(session, simulated=False)
    manifest = load_manifest(sdir, session, simulated=False)
    steps = select_steps(session, only)
    print(describe_session(session) if only is None else f"Session {session}: redo of step {only}.")
    print(PREFLIGHT)
    if input("\nPress Enter when the room is set up, 'q' to quit: ").strip().lower() == "q":
        return
    for i, step in enumerate(steps, 1):
        while True:
            print(f"\nStep {i}/{len(steps)}: {step['label']} ({step['seconds']} s of sound)")
            print(textwrap.fill(step["text"], 90, initial_indent="  ", subsequent_indent="  "))
            done = current(manifest, step["label"])
            if done and only is not None:
                # an explicit --step run is a redo: go straight to recording
                print(f"  Previous take ({done['stem']}) will be marked superseded: {format_check(done['qc'])}")
            elif done:
                print(f"  Already recorded ({done['stem']}): {format_check(done['qc'])}")
                ans = input("  Enter to keep it, 'r' to record it again, 'q' to quit: ").strip().lower()
                if ans == "q":
                    return
                if ans != "r":
                    break
            if step.get("away"):
                away = int(np.ceil(step["countdown"] + cfg.lead_s + cfg.sc_nsyms / cfg.sc_f_sym
                                   + step["seconds"] + cfg.tail_s)) + 5
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
            meta = capture_meta(step, session, cfg, layout, dev)
            stem = save_capture(sdir, f"s{session}_{step['label']}", rec, cfg.fs, meta)
            qc = quick_check(rec, meta)
            add_capture(manifest, step, stem, qc)
            save_manifest(sdir, manifest)
            print(f"  saved {stem.name}: {format_check(qc)}")
            if input("  Enter to continue, 'r' to redo this step: ").strip().lower() != "r":
                break
    print_summary(manifest)


# --------------------------------------------------------------------- check / summary
def check(session: int, simulated: bool) -> None:
    sdir = session_dir(session, simulated)
    manifest = load_manifest(sdir, session, simulated)
    if not manifest["captures"]:
        sys.exit(f"no captures in {sdir}")
    for c in manifest["captures"]:
        if not c["superseded"]:
            rec, meta = load_capture(sdir / c["stem"])
            c["qc"] = quick_check(rec, meta)
    save_manifest(sdir, manifest)
    print_summary(manifest)


def print_summary(manifest: dict) -> None:
    print(f"\nSession {manifest['session']}{' (simulated)' if manifest.get('simulated') else ''}:")
    print(f"  {'step':20s} {'class':7s} {'dist':>4s} {'cue':>3s} {'motion dB':>11s} {'rate/min':>9s}")
    missing = []
    for step in session_steps(manifest["session"]):
        c = current(manifest, step["label"])
        if c is None:
            missing.append(step["label"])
            continue
        chans = c["qc"]["channels"]
        motion = "/".join(f"{x['motion_db']:.1f}" if "motion_db" in x else "err" for x in chans)
        rate = "/".join(f"{x['rate_bpm']:.1f}" for x in chans if "rate_bpm" in x) or "-"
        dist = f"{c['distance_m']:.1f}" if c["distance_m"] else "-"
        cue = f"{c['cue_bpm']:.0f}" if c["cue_bpm"] else "-"
        warn = "\n    ! " + "; ".join(c["qc"]["warnings"]) if c["qc"]["warnings"] else ""
        print(f"  {step['label']:20s} {step['cls']:7s} {dist:>4s} {cue:>3s} {motion:>11s} {rate:>9s}{warn}")
    print("  complete" if not missing else f"  missing: {', '.join(missing)}")


# --------------------------------------------------------------------------- simulate
def simulate(session: int, cfg: SounderConfig, only: str | None = None) -> None:
    """Write simulated captures for every step of a session, or one step (no audio)."""
    from radar.scenes import scene_paths
    from radar.sim import propagate
    sdir = session_dir(session, simulated=True)
    manifest = load_manifest(sdir, session, simulated=True)
    print(describe_session(session))
    labels = [s["label"] for s in select_steps(session, only)]
    for i, step in enumerate(session_steps(session)):
        if step["label"] not in labels:
            continue
        tx, layout = build_tx(cfg, step["seconds"], cue_bpm=step.get("cue"))
        seed = 100 * session + i
        rec = np.column_stack([propagate(tx, cfg.fs, scene_paths(step, ch, session, seed),
                                         noise_std=2e-3, seed=10 * seed + ch) for ch in (0, 1)])
        meta = capture_meta(step, session, cfg, layout, {"simulated": True})
        stem = save_capture(sdir, f"s{session}_{step['label']}", rec, cfg.fs, meta)
        qc = quick_check(rec, meta)
        add_capture(manifest, step, stem, qc)
        save_manifest(sdir, manifest)
        print(f"  {step['label']:22s} {format_check(qc)}")
    print_summary(manifest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", type=int, required=True, choices=range(1, N_SESSIONS + 1))
    ap.add_argument("--check", action="store_true", help="re-run the quality check offline (no sound)")
    ap.add_argument("--simulate", action="store_true", help="simulated captures (no sound)")
    ap.add_argument("--step", help="record (or simulate) only this step label, e.g. breath_100cm_10bpm")
    ap.add_argument("--amplitude", type=float, default=0.5)
    args = ap.parse_args()
    cfg = SounderConfig(amplitude=args.amplitude)
    if args.check:
        check(args.session, args.simulate)
    elif args.simulate:
        simulate(args.session, cfg, args.step)
    else:
        record(args.session, cfg, args.step)


if __name__ == "__main__":
    main()
