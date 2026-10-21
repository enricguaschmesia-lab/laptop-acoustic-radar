"""Simultaneous play/record on the laptop's own speaker and microphone (one audio clock),
and saving captures as WAV + JSON sidecar."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

WASAPI = "Windows WASAPI"
# WASAPI shared mode on Windows (stable latency, no resampling surprises); the system
# default devices elsewhere.
DEFAULT_HOSTAPI = WASAPI if sys.platform == "win32" else None


def _hostapi_index(name: str) -> int:
    import sounddevice as sd
    for i, h in enumerate(sd.query_hostapis()):
        if h["name"] == name:
            return i
    raise RuntimeError(f"host API {name!r} not found")


def default_devices(hostapi: str | None = DEFAULT_HOSTAPI) -> tuple[int, int]:
    import sounddevice as sd
    if hostapi is None:
        in_dev, out_dev = sd.default.device
        return (in_dev if in_dev >= 0 else sd.query_devices(kind="input")["index"],
                out_dev if out_dev >= 0 else sd.query_devices(kind="output")["index"])
    h = sd.query_hostapis(_hostapi_index(hostapi))
    return h["default_input_device"], h["default_output_device"]


def playrec(tx: np.ndarray, fs: int, hostapi: str | None = DEFAULT_HOSTAPI, in_channels: int = 2,
            exclusive: bool = False) -> tuple[np.ndarray, dict]:
    """Play tx on both speaker channels while recording; returns (rec[n, ch], device info).

    Speaker and microphone run on one audio clock, so the round trip has a fixed latency
    and no sample-rate drift: that is what makes the laptop usable as a coherent radar.
    """
    import sounddevice as sd
    in_dev, out_dev = default_devices(hostapi)
    out = np.column_stack([tx, tx]).astype(np.float32)
    extra = None
    if exclusive and hostapi == WASAPI:
        extra = (sd.WasapiSettings(exclusive=True), sd.WasapiSettings(exclusive=True))
    rec = sd.playrec(out, samplerate=fs, channels=in_channels, dtype="float32",
                     device=(in_dev, out_dev), extra_settings=extra, blocking=True)
    status = sd.get_status()   # dropped buffers would corrupt the timing of the capture
    info = {"hostapi": hostapi or "default", "exclusive": exclusive,
            "input": sd.query_devices(in_dev)["name"], "output": sd.query_devices(out_dev)["name"],
            "input_overflow": bool(status.input_overflow), "input_underflow": bool(status.input_underflow),
            "output_underflow": bool(status.output_underflow), "output_overflow": bool(status.output_overflow)}
    return np.asarray(rec), info


def countdown(seconds: int) -> None:
    """Print a one-line countdown so the operator can take position before the sound starts."""
    for s in range(seconds, 0, -1):
        print(f"  starting in {s} s ...", end="\r", flush=True)
        time.sleep(1)
    print(" " * 30, end="\r")


def save_capture(directory: Path, label: str, rec: np.ndarray, fs: int, meta: dict) -> Path:
    """Write <timestamp>_<label>.wav (float32, all mic channels) and a .json sidecar."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / f"{time.strftime('%Y%m%d-%H%M%S')}_{label}"
    wavfile.write(stem.with_suffix(".wav"), fs, rec.astype(np.float32))
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return stem


def load_capture(stem: Path) -> tuple[np.ndarray, dict]:
    stem = Path(stem).with_suffix("")
    fs, rec = wavfile.read(stem.with_suffix(".wav"))
    meta = json.loads(stem.with_suffix(".json").read_text())
    rec = rec if rec.ndim == 2 else rec[:, None]
    return rec.astype(np.float64), meta
