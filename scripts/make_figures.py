"""Figures for the README (docs/figures/), drawn from the local recordings and saved results.

Needs the raw dataset in data/raw/dataset (not published) plus results/ml/metrics.json,
results/stress/results.json and results/breathing/check.json. The Part 1 figures in
docs/figures/ofdm_* come from the EE-442 report and are not regenerated here.

    python scripts/make_figures.py            # all figures
    python scripts/make_figures.py --no-gif   # skip the animation
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import uniform_filter

from radar.audio_io import load_capture
from radar.config import SounderConfig
from radar.processing import (
    arc_phase,
    breathing_rate,
    doppler_to_speed,
    pca_projection,
    process_recording,
    range_axis,
    range_doppler,
    remove_static,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
DATA = ROOT / "data" / "raw" / "dataset"

BG, NAVY, TEAL, AMBER, GREY, SOFT = "#F7F5F0", "#14213D", "#1F7A8C", "#D9731A", "#8A94A6", "#3D4A5C"
MAP = LinearSegmentedColormap.from_list("radar", ["#14213D", "#1F4E79", "#1F7A8C", "#7FB8A4", "#F2D16B", "#FBF4DC"])

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 13, "axes.edgecolor": SOFT, "axes.labelcolor": NAVY, "xtick.color": SOFT,
    "ytick.color": SOFT, "text.color": NAVY, "axes.titlecolor": NAVY, "figure.facecolor": BG,
    "axes.facecolor": BG, "savefig.facecolor": BG, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def capture(session: int, label: str):
    sdir = DATA / f"session_{session}"
    m = json.loads((sdir / "manifest.json").read_text())
    c = [c for c in m["captures"] if c["label"] == label and not c["superseded"]][-1]
    rec, meta = load_capture(sdir / c["stem"])
    return rec, meta, SounderConfig.from_dict(meta["cfg"])


def responses(session: int, label: str, mic: int = 0):
    rec, meta, cfg = capture(session, label)
    return process_recording(rec[:, mic], cfg, meta["layout"])["h"], cfg


def motion_map_db(h: np.ndarray, cfg: SounderConfig, smooth=(24, 3)) -> np.ndarray:
    """Clutter-removed power in dB re the total received power, smoothed over 0.4 s x 3 bins."""
    p = np.abs(remove_static(h, cfg.frame_rate, window_s=2.0)) ** 2
    total = np.mean(np.sum(np.abs(h) ** 2, axis=1))
    return 10 * np.log10(uniform_filter(p, size=smooth) / total + 1e-12)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("  ", name)


def hero() -> None:
    """The four activity classes as the radar sees them: range over time, same colour scale."""
    scenes = [(1, "empty", "Empty room"), (1, "breath_40cm_10bpm", "Breathing, 40 cm"),
              (1, "waving", "Hand waving"), (1, "walking", "Walking, 0.8-2.5 m")]
    t0, dur = 8.0, 14.0
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.3), sharey=True, layout="constrained")
    for a, (s, label, title) in zip(axs, scenes, strict=True):
        h, cfg = responses(s, label)
        db = motion_map_db(h, cfg)
        r = range_axis(cfg)
        i0, i1 = int(t0 * cfg.frame_rate), int((t0 + dur) * cfg.frame_rate)
        im = a.imshow(db[i0:i1].T, aspect="auto", origin="lower", extent=[0, dur, r[0], r[-1]],
                      vmin=-62, vmax=-22, cmap=MAP, interpolation="bilinear")
        a.set(title=title, xlabel="time (s)", ylim=(0, 2.6))
        a.spines[:].set_visible(False)
    axs[0].set_ylabel("range (m)")
    cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.01, aspect=30)
    cb.set_label("motion power (dB)")
    cb.outline.set_visible(False)
    save(fig, "hero_activities.png")


def walking() -> None:
    h, cfg = responses(1, "walking")
    r = range_axis(cfg)
    t = np.arange(len(h)) / cfg.frame_rate
    db = motion_map_db(h, cfg)
    lo, hi = np.percentile(db, [55, 99.7])
    maps, fd, tw = range_doppler(remove_static(h, cfg.frame_rate), cfg)
    md = 10 * np.log10(maps.sum(axis=2) + 1e-20)
    v = doppler_to_speed(fd, cfg)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True)
    ax[0].imshow(db.T, aspect="auto", origin="lower", extent=[0, t[-1], r[0], r[-1]], vmin=lo, vmax=hi, cmap=MAP)
    ax[0].set(ylabel="range (m)", ylim=(0, 2.6), title="Where: range over time")
    ax[1].imshow(md.T, aspect="auto", origin="lower", extent=[tw[0], tw[-1], v[0], v[-1]],
                 vmin=md.max() - 38, vmax=md.max(), cmap=MAP)
    ax[1].set(ylabel="speed (m/s)", xlabel="time (s)", title="How fast: radial speed over time (micro-Doppler)")
    fig.tight_layout()
    save(fig, "walking_range_doppler.png")


def breathing_wrap() -> None:
    h, cfg = responses(1, "breath_40cm_10bpm")
    tap = 4 + 11          # the 47 cm bin used in the session-1 analysis
    z = h[:, tap]
    t = np.arange(len(z)) / cfg.frame_rate
    zc = z - z.mean()
    proj = pca_projection(z)
    disp = arc_phase(z) * cfg.wavelength / (4 * np.pi) * 100     # cm
    fig = plt.figure(figsize=(14, 4.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2.2], hspace=0.6, wspace=0.18)
    a0 = fig.add_subplot(gs[:, 0])
    a0.plot(zc.real, zc.imag, color=TEAL, lw=0.8)
    a0.set(aspect="equal", xlabel="I", ylabel="Q", title="Echo at 47 cm (I/Q)")
    a0.set_xticks([])
    a0.set_yticks([])
    a1 = fig.add_subplot(gs[0, 1])
    a1.plot(t, proj / np.abs(proj).max(), color=AMBER, lw=1.6)
    a1.set(title="Straight-line projection: two humps per breath, reads 20.3/min", yticks=[])
    a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    a2.plot(t, disp - disp.mean(), color=TEAL, lw=1.6)
    a2.set(title="Angle around the fitted circle: one cycle per breath, reads 10.2/min",
           xlabel="time (s)", ylabel="cm")
    save(fig, "breathing_phase_wrap.png")


def breathing_tone() -> None:
    from breathing_check import pitch_track
    rec, meta, cfg = capture(4, "breath_70cm_15bpm")
    p = process_recording(rec[:, 0], cfg, meta["layout"])
    est = breathing_rate(p["h"], cfg, min_range=0.15, max_range=1.6)
    skip = int(np.ceil(0.3 * cfg.frame_rate))
    centres = p["start"] + (skip + np.arange(len(p["h"]))) * cfg.period + cfg.period // 2
    pitch = pitch_track(rec[:, 0], centres, cfg.fs)
    t = np.arange(len(pitch)) / cfg.frame_rate
    z = lambda v: (v - v.mean()) / v.std()
    sig = est["signal"]
    sgn = np.sign(np.corrcoef(sig, pitch)[0, 1]) or 1
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(t, z(pitch), color=GREY, lw=2.0, label="pacing tone (pitch, from the recording)")
    ax.plot(t, sgn * z(sig), color=TEAL, lw=1.8, label="radar breathing signal")
    ax.set(xlabel="time (s)", yticks=[],
           title=f"Held-out session 4, 70 cm, cue 15/min: radar reads {est['rate_bpm']:.2f}/min")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False)
    fig.tight_layout()
    save(fig, "breathing_vs_tone.png")


def breathing_deviation() -> None:
    rows = json.loads((ROOT / "results" / "breathing" / "check.json").read_text())["rows"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for sess, col, lab in (((1, 2), GREY, "sessions 1-2 (tuning)"), ((3, 4), TEAL, "sessions 3-4 (held out)")):
        sel = [r for r in rows if r["session"] in sess]
        x = (np.array([r["cue_bpm"] for r in sel]) + (np.array([r["mic"] for r in sel]) - 0.5) * 0.7
             + (0.35 if 3 in sess else -0.35))
        ax.scatter(x, [r["radar_offset"] for r in sel], s=80, color=col, label=lab, zorder=3)
    ax.axhspan(-0.5, 0.5, color=TEAL, alpha=0.08, zorder=0)
    ax.axhline(0, color=SOFT, lw=1)
    ax.set(xticks=[10, 15, 20], xlabel="cue (breaths/min)", ylabel="estimate minus cue (/min)",
           ylim=(-1.0, 1.0), title="Deviation from the cue, both mics")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    save(fig, "breathing_deviation.png")


def confusion() -> None:
    m = json.loads((ROOT / "results" / "ml" / "metrics.json").read_text())
    classes = m["classes"]
    cmap = LinearSegmentedColormap.from_list("b", [BG, TEAL, NAVY])
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.9))
    for a, key, title in zip(axs, ("energy_lr", "features_rf", "cnn"),
                             ("Energy only", "Features + random forest", "Small CNN"), strict=True):
        cm = np.asarray(m["activity"][key]["confusion"], float)
        cmn = cm / cm.sum(axis=1, keepdims=True)
        a.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                a.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=14,
                       color=BG if cmn[i, j] > 0.55 else NAVY)
        a.set(xticks=range(len(classes)), xticklabels=classes, yticks=range(len(classes)), yticklabels=classes,
              xlabel="predicted", title=f"{title}\nmacro-F1 {m['activity'][key]['pooled_macro_f1']:.3f}")
        a.spines[:].set_visible(False)
    axs[0].set_ylabel("true")
    fig.tight_layout()
    save(fig, "confusion.png")


def stress() -> None:
    r = json.loads((ROOT / "results" / "stress" / "results.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
    for a, win in zip(axs, ("4s", "1s"), strict=True):
        rows = sorted((v for v in r.values() if v["window"] == win),
                      key=lambda v: -50 if v["level_db"] is None else v["level_db"])
        x = [-35 if v["level_db"] is None else v["level_db"] for v in rows]
        for key, col, lab, ls in (("cnn", AMBER, "Small CNN", "-"), ("features_rf", TEAL, "Features + RF", "-"),
                                  ("features_lr", TEAL, "Features + LR", "--"),
                                  ("energy_lr", GREY, "Energy only", "-")):
            vals = np.array([list(v[key].values()) for v in rows])
            a.plot(x, vals.mean(axis=1), ls, color=col, lw=2.4, marker="o", ms=6, label=lab)
            a.fill_between(x, vals.min(axis=1), vals.max(axis=1), color=col, alpha=0.12)
        for name, lvl in (("still", -22.7), ("walking", -17.8)):
            a.axvline(lvl, color=SOFT, ls=":", lw=1.2)
            a.text(lvl + 0.4, 0.05, f"{name} signal", rotation=90, fontsize=12, color=SOFT, va="bottom")
        a.set_xticks([-35, -30, -20, -15, -10, -5, 0], ["none", "-30", "-20", "-15", "-10", "-5", "0"])
        a.set(ylim=(0, 1.05), xlabel="added noise (dB re total received power)",
              title=f"{win[0]} s windows ({'256' if win == '4s' else '64'} snapshots)")
    axs[0].set_ylabel("macro-F1, held-out sessions")
    axs[0].legend(loc="lower left", frameon=False)
    fig.tight_layout()
    save(fig, "stress_noise.png")


def animation(seconds: float = 16.0, fps: int = 8) -> None:
    """Live view of the walking capture: range profile now (left) and range over time (right)."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    h, cfg = responses(1, "walking")
    r = range_axis(cfg)
    db = motion_map_db(h, cfg)
    t0 = 6.0
    i0 = int(t0 * cfg.frame_rate)
    n = int(seconds * cfg.frame_rate)
    db = db[i0:i0 + n]
    step = int(round(cfg.frame_rate / fps))
    frames = range(step, n, step)
    lo, hi = np.percentile(db, [55, 99.7])
    sel = r <= 2.6

    with plt.rc_context({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                         "xtick.labelsize": 10, "ytick.labelsize": 10}):
        fig = plt.figure(figsize=(8.0, 3.6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 3.2], wspace=0.06)
        a0 = fig.add_subplot(gs[0])
        a1 = fig.add_subplot(gs[1], sharey=a0)
        (line,) = a0.plot(db[0, sel], r[sel], color=AMBER, lw=2)
        a0.set(xlim=(lo, hi), ylim=(0, 2.6), xlabel="motion power", ylabel="range (m)", title="Now")
        a0.set_xticks([])
        img = np.full_like(db, np.nan)
        im = a1.imshow(img.T, aspect="auto", origin="lower", extent=[0, seconds, r[0], r[-1]],
                       vmin=lo, vmax=hi, cmap=MAP.resampled(32))     # few colours keep the GIF small
        cursor = a1.axvline(0, color=AMBER, lw=1.5)
        a1.set(xlim=(0, seconds), xlabel="time (s)", title="Walking towards and away from the laptop")
        plt.setp(a1.get_yticklabels(), visible=False)
        for a in (a0, a1):
            a.spines[:].set_visible(False)
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.88)

        def draw(k):
            img[:k] = db[:k]
            im.set_data(img.T)
            line.set_xdata(np.clip(db[k - 1, sel], lo, hi))
            cursor.set_xdata([k / cfg.frame_rate] * 2)
            return im, line, cursor

        anim = FuncAnimation(fig, draw, frames=frames, blit=False)
        OUT.mkdir(parents=True, exist_ok=True)
        anim.save(OUT / "walking_live.gif", writer=PillowWriter(fps=fps), dpi=90)
        plt.close(fig)
    shared_palette(OUT / "walking_live.gif", fps)
    print("   walking_live.gif")


def shared_palette(path: Path, fps: int, colors: int = 64) -> None:
    """Re-encode a GIF with one palette for all frames: about 5x smaller than per-frame palettes."""
    from PIL import Image, ImageSequence
    frames = [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(path))]
    # palette from the last frame (it has every map colour) plus a block of the accent colour
    src = Image.new("RGB", (frames[-1].width, frames[-1].height + 40), AMBER)
    src.paste(frames[-1], (0, 0))
    pal = src.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=round(1000 / fps), loop=0, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()
    hero()
    walking()
    breathing_wrap()
    breathing_tone()
    breathing_deviation()
    confusion()
    stress()
    if not args.no_gif:
        animation()


if __name__ == "__main__":
    main()
