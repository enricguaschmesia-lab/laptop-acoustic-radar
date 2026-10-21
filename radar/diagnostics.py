"""Hardware go/no-go diagnostics for a static scene (nobody moving)."""

import numpy as np
from scipy.signal import welch

from .config import SounderConfig
from .processing import GUARD_TAPS, process_recording, range_axis, remove_static

# Pass thresholds for the static check (judgement calls, stated in the report)
MIN_SNR_DB = 25.0          # in-band level during sounding vs the lead silence
MAX_AMP_DRIFT = 0.10       # direct-path amplitude change, first vs last second
MAX_AMP_STD = 0.05         # relative std of the direct-path amplitude
MAX_PHASE_STD_DEG = 10.0   # direct-path phase jitter after removing a linear trend
MAX_FLOOR_DB = -40.0       # static-scene residual (clutter-removed) vs direct-path power


def analyze_static(x: np.ndarray, layout: dict, cfg: SounderConfig) -> dict:
    fs = cfg.fs
    res = {"peak": float(np.max(np.abs(x))), "rms": float(np.sqrt(np.mean(x ** 2))),
           "clipped": int(np.sum(np.abs(x) >= 0.999))}

    out = process_recording(x, cfg, layout)
    h = out["h"]
    res["latency_ms"] = 1e3 * out["latency_samples"] / fs
    res["sync_peak_over_threshold"] = out["sync"]["peak_val"] / out["sync"]["threshold"]

    # In-band SNR at the microphone: sounding vs the silence before the preamble
    noise = x[int(0.05 * fs): int((cfg.lead_s - 0.05) * fs)]
    s0 = out["start"] + int(0.3 * fs)
    sound = x[s0: out["start"] + layout["n_periods"] * cfg.period]
    f, p_noise = welch(noise, fs, nperseg=4096)
    _, p_sound = welch(sound, fs, nperseg=4096)
    band = (f >= cfg.f_low) & (f <= cfg.f_low + cfg.bandwidth)
    res["inband_snr_db"] = float(10 * np.log10(p_sound[band].mean() / p_noise[band].mean()))
    res["psd"] = (f, p_noise, p_sound)

    # Speaker + mic frequency response across the sounding band
    H_mag_db = 20 * np.log10(np.abs(out["H"]).mean(axis=0))
    res["H_spread_db"] = float(H_mag_db.max() - H_mag_db.min())
    res["H_mag_db"] = H_mag_db

    # Direct-path stability over the capture
    d = h[:, GUARD_TAPS]
    amp = np.abs(d)
    one_s = int(cfg.frame_rate)
    res["amp_rel_std"] = float(amp.std() / amp.mean())
    res["amp_drift"] = float(amp[-one_s:].mean() / amp[:one_s].mean() - 1)
    phase = np.unwrap(np.angle(d))
    t = np.arange(len(d)) / cfg.frame_rate
    slope, icpt = np.polyfit(t, phase, 1)
    res["phase_slope_deg_s"] = float(np.degrees(slope))
    res["phase_std_deg"] = float(np.degrees(np.std(phase - (slope * t + icpt))))
    res["direct_amp"], res["direct_phase"], res["t"] = amp, phase, t

    # Residual after static clutter removal: the floor any moving target must beat
    h_dyn = remove_static(h, cfg.frame_rate)
    r = range_axis(cfg)
    sel = (r >= 0.15) & (r <= 2.0)
    direct_power = np.mean(np.abs(d) ** 2)
    res["floor_db"] = float(10 * np.log10(np.median(np.mean(np.abs(h_dyn[:, sel]) ** 2, axis=0)) / direct_power))
    res["pdp_db"] = 10 * np.log10(np.mean(np.abs(h) ** 2, axis=0) / direct_power)
    res["h_dyn_db"] = 10 * np.log10(np.abs(h_dyn) ** 2 / direct_power + 1e-12)
    res["range"] = r

    checks = {
        "no clipping": res["clipped"] == 0,
        f"in-band SNR >= {MIN_SNR_DB:.0f} dB": res["inband_snr_db"] >= MIN_SNR_DB,
        f"direct-path drift <= {MAX_AMP_DRIFT:.0%}": abs(res["amp_drift"]) <= MAX_AMP_DRIFT,
        f"direct-path amplitude std <= {MAX_AMP_STD:.0%}": res["amp_rel_std"] <= MAX_AMP_STD,
        f"direct-path phase jitter <= {MAX_PHASE_STD_DEG:.0f} deg": res["phase_std_deg"] <= MAX_PHASE_STD_DEG,
        f"static floor <= {MAX_FLOOR_DB:.0f} dB": res["floor_db"] <= MAX_FLOOR_DB,
    }
    res["checks"] = checks
    res["pass"] = all(checks.values())
    return res


SUMMARY_KEYS = ["peak", "rms", "clipped", "latency_ms", "sync_peak_over_threshold", "inband_snr_db",
                "H_spread_db", "amp_rel_std", "amp_drift", "phase_std_deg", "phase_slope_deg_s", "floor_db"]


def summary(res: dict) -> dict:
    s = {k: res[k] for k in SUMMARY_KEYS}
    s["checks"] = res["checks"]
    s["pass"] = res["pass"]
    return s


def plot_static(res: dict, cfg: SounderConfig, title: str, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    f, pn, ps = res["psd"]
    ax[0, 0].semilogy(f / 1e3, pn, label="lead silence")
    ax[0, 0].semilogy(f / 1e3, ps, label="sounding")
    ax[0, 0].axvspan(cfg.f_low / 1e3, (cfg.f_low + cfg.bandwidth) / 1e3, alpha=0.15)
    ax[0, 0].set(xlabel="Frequency (kHz)", ylabel="PSD", xlim=(0, 16),
                 title=f"Mic spectrum, in-band SNR {res['inband_snr_db']:.1f} dB")
    ax[0, 0].legend()

    ax[0, 1].plot(res["range"] * 100, res["pdp_db"], marker=".")
    ax[0, 1].set(xlabel="Range from direct path (cm)", ylabel="dB re direct path",
                 title="Mean power-delay profile")
    ax[0, 1].grid(True)

    ax2 = ax[1, 0].twinx()
    ax[1, 0].plot(res["t"], res["direct_amp"] / res["direct_amp"].mean(), color="C0")
    ax2.plot(res["t"], np.degrees(res["direct_phase"] - res["direct_phase"][0]), color="C1")
    ax[1, 0].set(xlabel="Time (s)", ylabel="Direct-path amplitude (norm.)",
                 title=f"Direct path: drift {res['amp_drift']:+.1%}, phase jitter {res['phase_std_deg']:.1f} deg")
    ax2.set_ylabel("Phase change (deg)", color="C1")

    im = ax[1, 1].imshow(res["h_dyn_db"].T, aspect="auto", origin="lower", vmin=-70, vmax=-20,
                         extent=[0, res["t"][-1], res["range"][0] * 100, res["range"][-1] * 100], cmap="viridis")
    ax[1, 1].set(xlabel="Time (s)", ylabel="Range (cm)",
                 title=f"Clutter-removed range-time map, floor {res['floor_db']:.1f} dB")
    fig.colorbar(im, ax=ax[1, 1], label="dB re direct")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
