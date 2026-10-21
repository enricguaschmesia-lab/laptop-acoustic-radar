"""Is the small upward bias of the breathing estimates the estimator or the person?

1. The pacing tone (pitch gliding 440 -> 660 -> 440 Hz once per breath) is audible in every
   paced recording. Its pitch, tracked once per sounding period, is a periodic signal at
   exactly the cue rate, so running the same spectral peak estimator on it tests the time
   base and the estimator together, on the same frames as the radar signal.
2. The phase of the breathing signal against the tone's phase over each capture: a steady
   drift means the breathing ran at a different rate from the tone, whatever the estimator.

    python scripts/breathing_check.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from radar.audio_io import load_capture
from radar.config import SounderConfig
from radar.processing import _peak_freq, _spectrum, breathing_rate, process_recording
from radar.protocol import N_SESSIONS

ROOT = Path(__file__).resolve().parents[1]

BAND = (0.1, 0.7)
PITCH_BAND = (400.0, 700.0)
PITCH_WIN, PITCH_NFFT = 4096, 8192
EDGE_S = 4.0            # drop filter transients at both ends before fitting the drift


def pitch_track(x: np.ndarray, centres: np.ndarray, fs: int) -> np.ndarray:
    """Dominant frequency in PITCH_BAND around each sample index (Hann window, parabolic peak)."""
    half = PITCH_WIN // 2
    f = np.fft.rfftfreq(PITCH_NFFT, 1 / fs)
    sel = np.flatnonzero((f >= PITCH_BAND[0]) & (f <= PITCH_BAND[1]))
    taper = np.hanning(PITCH_WIN)
    xp = np.pad(x, half)
    out = np.empty(len(centres))
    for a in range(0, len(centres), 256):
        idx = centres[a:a + 256, None] + np.arange(PITCH_WIN)        # padded indexing: centre - half .. + half
        spec = np.abs(np.fft.rfft(xp[idx] * taper, PITCH_NFFT, axis=1))[:, sel]
        k = np.clip(np.argmax(spec, axis=1), 1, len(sel) - 2)
        r = np.arange(len(k))
        y0, y1, y2 = spec[r, k - 1], spec[r, k], spec[r, k + 1]
        denom = np.where(y0 - 2 * y1 + y2 == 0, -1e-12, y0 - 2 * y1 + y2)
        out[a:a + 256] = f[sel][k] + 0.5 * (y0 - y2) / denom * (f[1] - f[0])
    return out


def phase_drift_bpm(breath: np.ndarray, pitch: np.ndarray, f_cue: float, fs: float):
    """Rate offset (breaths/min) from the slope of the breathing-minus-tone phase."""
    sos = butter(2, [0.6 * f_cue, 1.5 * f_cue], btype="band", fs=fs, output="sos")
    ph_b = np.unwrap(np.angle(hilbert(sosfiltfilt(sos, breath - breath.mean()))))
    ph_c = np.unwrap(np.angle(hilbert(sosfiltfilt(sos, pitch - pitch.mean()))))
    dphi = ph_b - ph_c
    t = np.arange(len(dphi)) / fs
    keep = (t > EDGE_S) & (t < t[-1] - EDGE_S)
    slope = np.polyfit(t[keep], dphi[keep], 1)[0]
    return 60 * slope / (2 * np.pi), t, dphi - dphi[keep][0]


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    rows, traces = [], []
    for s in range(1, N_SESSIONS + 1):
        sdir = ROOT / "data" / "raw" / "dataset" / f"session_{s}"
        manifest = json.loads((sdir / "manifest.json").read_text())
        for c in manifest["captures"]:
            if c["superseded"] or not c["cue_bpm"]:
                continue
            rec, meta = load_capture(sdir / c["stem"])
            cfg = SounderConfig.from_dict(meta["cfg"])
            cue = float(c["cue_bpm"])
            for ch in range(rec.shape[1]):
                p = process_recording(rec[:, ch], cfg, meta["layout"])
                h = p["h"]
                est = breathing_rate(h, cfg, min_range=0.15, max_range=1.6)
                skip = int(np.ceil(0.3 * cfg.frame_rate))      # as in channel_estimates
                centres = p["start"] + (skip + np.arange(len(h))) * cfg.period + cfg.period // 2
                pitch = pitch_track(rec[:, ch], centres, cfg.fs)
                f, spec = _spectrum(pitch, cfg.frame_rate)
                cue_measured = 60 * _peak_freq(f, spec, BAND)
                drift, t, dphi = phase_drift_bpm(est["signal"], pitch, cue / 60, cfg.frame_rate)
                rows.append({"session": s, "label": c["label"], "mic": ch, "cue_bpm": cue,
                             "tone_rate_from_audio": cue_measured, "radar_rate": est["rate_bpm"],
                             "radar_offset": est["rate_bpm"] - cue, "phase_drift_offset": drift,
                             "method": est["method"], "range_m": est["range_m"]})
                traces.append((s, c["label"], ch, cue, t, dphi, pitch, est["signal"]))

    tone_err = np.array([r["tone_rate_from_audio"] - r["cue_bpm"] for r in rows])
    radar_off = np.array([r["radar_offset"] for r in rows])
    drift_off = np.array([r["phase_drift_offset"] for r in rows])
    summary = {
        "n": len(rows),
        "tone_from_audio_mean_error": float(tone_err.mean()),
        "tone_from_audio_max_abs_error": float(np.abs(tone_err).max()),
        "radar_offset_mean": float(radar_off.mean()), "phase_drift_offset_mean": float(drift_off.mean()),
        "radar_vs_drift_correlation": float(np.corrcoef(radar_off, drift_off)[0, 1]),
        "radar_minus_drift_mean_abs": float(np.mean(np.abs(radar_off - drift_off))),
    }
    out = ROOT / "results" / "breathing"
    out.mkdir(parents=True, exist_ok=True)
    (out / "check.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    print(f"{'capture':24s} mic  cue  tone(audio)  radar   radar-cue  phase-drift")
    for r in rows:
        print(f"s{r['session']} {r['label']:21s} {r['mic']}  {r['cue_bpm']:4.0f}  {r['tone_rate_from_audio']:10.3f}  "
              f"{r['radar_rate']:6.2f}  {r['radar_offset']:+8.2f}  {r['phase_drift_offset']:+10.2f}")
    print("\n" + "\n".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in summary.items()))
    plot(rows, traces, out)


def plot(rows, traces, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    s, label, ch, cue, t, dphi, pitch, sig = traces[0]
    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    ax[0].plot(t, z(pitch), label="tone pitch (from the recording)")
    ax[0].plot(t, z(sig), label="breathing displacement (radar)")
    ax[0].set(xlabel="Time (s)", ylabel="normalised", title=f"s{s} {label}, mic {ch}")
    ax[0].legend(fontsize=8)
    cmap = {10.0: "C0", 15.0: "C1", 20.0: "C2"}
    for *_, cue, t, dphi, _, _ in traces:
        ax[1].plot(t, dphi / (2 * np.pi), color=cmap.get(cue, "k"), lw=0.8, alpha=0.8)
    for cue, col in cmap.items():
        ax[1].plot([], [], color=col, label=f"cue {cue:.0f}/min")
    ax[1].set(xlabel="Time (s)", ylabel="breathing minus tone phase (cycles)", title="Phase drift against the tone")
    ax[1].legend(fontsize=8)
    ro = [r["radar_offset"] for r in rows]
    do = [r["phase_drift_offset"] for r in rows]
    ax[2].scatter(do, ro, c=[cmap.get(r["cue_bpm"], "k") for r in rows])
    lim = max(map(abs, ro + do)) * 1.2 + 0.05
    ax[2].plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    ax[2].axhline(0, color="grey", lw=0.5)
    ax[2].axvline(0, color="grey", lw=0.5)
    ax[2].set(xlabel="rate offset from phase drift (/min)", ylabel="radar estimate minus cue (/min)",
              title="Two independent views of the offset", xlim=(-lim, lim), ylim=(-lim, lim))
    fig.tight_layout()
    fig.savefig(out / "breathing_check.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
