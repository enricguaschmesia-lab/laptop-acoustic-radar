"""Offline look at a saved capture (no audio is played): spectrogram, in-band and
out-of-band level over time, and how the impulse response evolves.

    python scripts/inspect_capture.py data/raw/smoke/<stem>
"""

import argparse
from pathlib import Path

import numpy as np

from radar.audio_io import load_capture
from radar.config import SounderConfig
from radar.processing import GUARD_TAPS, process_recording, range_axis

ROOT = Path(__file__).resolve().parents[1]


def band_envelope(x, fs, lo, hi, frame=768):
    n = len(x) // frame
    spec = np.abs(np.fft.rfft(x[: n * frame].reshape(n, frame), axis=1)) ** 2
    f = np.fft.rfftfreq(frame, 1 / fs)
    sel = (f >= lo) & (f < hi)
    return np.arange(n) * frame / fs, 10 * np.log10(spec[:, sel].mean(axis=1) + 1e-20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rec, meta = load_capture(Path(args.stem))
    cfg = SounderConfig.from_dict(meta["cfg"])
    layout = meta["layout"]
    fs = cfg.fs
    n_ch = rec.shape[1]

    fig, ax = plt.subplots(n_ch, 3, figsize=(16, 4.2 * n_ch), squeeze=False)
    for ch in range(n_ch):
        x = rec[:, ch]
        ax[ch, 0].specgram(x, NFFT=2048, Fs=fs, noverlap=1024, cmap="magma", vmin=-160)
        ax[ch, 0].set(ylim=(0, 16000), xlabel="Time (s)", ylabel="Hz", title=f"ch{ch} spectrogram")

        for lo, hi, name in [(cfg.f_low, cfg.f_low + cfg.bandwidth, "in-band 6-10 kHz"),
                             (2000, 5000, "2-5 kHz"), (11000, 15000, "11-15 kHz")]:
            t, env = band_envelope(x, fs, lo, hi)
            ax[ch, 1].plot(t, env, label=name)
        ax[ch, 1].axvline(layout["sound_start"] / fs, color="k", ls="--", lw=0.8)
        ax[ch, 1].set(xlabel="Time (s, recording)", ylabel="dB",
                      title="Band levels per 16 ms (dashed: TX sounding start)")
        ax[ch, 1].legend(fontsize=8)

        try:
            out = process_recording(x, cfg, layout)
            h = out["h"]
            t_h = np.arange(len(h)) / cfg.frame_rate
            r = range_axis(cfg)
            for tap in [GUARD_TAPS, GUARD_TAPS + 2, GUARD_TAPS + 6, GUARD_TAPS + 12, GUARD_TAPS + 24]:
                ax[ch, 2].plot(t_h, 20 * np.log10(np.abs(h[:, tap]) + 1e-12), label=f"{r[tap] * 100:.0f} cm")
            total = 10 * np.log10(np.sum(np.abs(out["H"]) ** 2, axis=1))
            ax[ch, 2].plot(t_h, total, "k", lw=1.2, label="total in-band")
            ax[ch, 2].set(xlabel="Time (s, after sync)", ylabel="dB",
                          title=f"Taps over time (latency {1e3 * out['latency_samples'] / fs:.0f} ms)")
            ax[ch, 2].legend(fontsize=8)
        except Exception as e:  # sync can fail on a broken capture
            ax[ch, 2].text(0.1, 0.5, f"processing failed: {e}", transform=ax[ch, 2].transAxes)

    fig.suptitle(Path(args.stem).name)
    fig.tight_layout()
    out_png = ROOT / "results" / "smoke" / f"{Path(args.stem).name}_inspect.png"
    fig.savefig(out_png, dpi=110)
    print(out_png)


if __name__ == "__main__":
    main()
