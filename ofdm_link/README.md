# Part 1: acoustic OFDM link (MATLAB)

Course project for EPFL **EE-442 Wireless Receivers: Algorithms and Architectures**, Group 20:
**Enric Guasch Mesià, Joelle Alachkar, Petar Damjanovic**.

A complete OFDM modem that sends random bits and images from a loudspeaker to a microphone
through the air. The full write-up, with every experiment and figure, is in
[`EE-442_acoustic_OFDM_report.pdf`](EE-442_acoustic_OFDM_report.pdf).

The code is published as submitted. The radar in the rest of this repository reuses its preamble,
pulse shaping and frame synchronisation through a Python port that is checked against this code
(`tests/fixtures/make_fixtures.m`, `tests/test_port.py`).

## System

| Parameter | Value |
|---|---|
| Audio | 48 kHz, carrier 8 kHz, 2 kHz band (7–9 kHz) |
| OFDM | 512 subcarriers (3.9 Hz spacing), QPSK with Gray mapping, 256-sample (128 ms) cyclic prefix |
| Frame | BPSK preamble + 1 training OFDM symbol + 50 data symbols (`txofdm1.m`: periodic training) |
| Preamble | 500 BPSK symbols at 1 kBd, RRC roll-off 0.22, PN from an order-11 LFSR (x¹¹ + x² + 1) |
| Sync | Normalised correlation against the known preamble, global peak (`frame_sync.m`) |
| Channel | Least squares per subcarrier, impulse response truncated after the last tap above 1 % of the peak |
| Tracking | Decision-directed per subcarrier, Viterbi–Viterbi common phase, per-subcarrier Viterbi–Viterbi |
| Peak power | XOR whitening of the payload (`XORING.m`): PAPR 28.0 → 13.0 dB |

## Files

`matlab/` holds the final submission.

| Entry script | What it does |
|---|---|
| `image_tx_orig_baseline.m`, `image_tx_orig_whitening.m` | Send `orig.png` over the link, without and with XOR whitening, and rebuild the image |
| `image_tx_viterbi_baseline.m`, `image_tx_viterbi_whitening.m` | Same with `viterbi.jpg` |
| `cool_graphs.m`, `save_fig_script.m` | One transmission plus channel, power-delay-profile, sync and constellation diagnostics |
| `ber_vs_ncp.m` | BER against cyclic-prefix length |
| `ber_vs_eta.m` | BER against spectral efficiency, sweeping the training interval |
| `papr_comparison.m` | Peak-to-average power: baseline, subcarrier interleaving, XOR whitening |

| Function | Role |
|---|---|
| `txofdm.m`, `txofdm1.m` | Transmitters: single training symbol, periodic training |
| `rxofdm.m`, `rxofdm1.m`, `rxofdm2.m` | Matching receivers; `rxofdm2.m` also returns the diagnostics used for plotting |
| `frame_sync.m`, `lfsr_framesync.m` | Frame synchroniser and PN sequence |
| `save_audio_ofdm_plots.m` | Exports channel, PDP, channel-evolution and constellation plots |
| `XORING.m` | Whitening and de-whitening sequence |
| `rrc.m`, `ofdmlowpass.m`, `ofdm_tx_resampling.m`, `ofdm_rx_resampling.m` | Pulse shaping, filtering and resampling (course framework) |

`matlab/tasks_123/` holds the code and figures of the first three course tasks: BER against SNR
on the emulator, phase tracking under oscillator drift, and channel characterisation.

## Running it

Run the entry scripts from `matlab/` (they read `orig.png` and `viterbi.jpg` from the working
folder). With `conf.audiosystem = 'audio'` they play through the default speaker and record the
default microphone. The course's channel emulator (`channel_emulator.p`) is course material and
is not included, so the `'emulator'` mode needs a copy from the course.

## Errata

Issues found when the project was revisited:

- The report says the BER is "< 1 % at −5.5 dB" on emulator channel 1; it is about 10 % there.
- The non-monotonic BER-vs-prefix sweep was blamed on inter-symbol interference, but even the
  shortest prefix (32 ms) was over ten times the 2.5 ms echo spread. The differences were noise
  between three hand-held trials.
- `frame_sync.m` computes a detection threshold but never uses it (the global peak is taken), so
  false-alarm and miss rates were never measured.
- The report describes the common-phase tracker as the angle of the mean normalised symbol; the
  code implements the 4th-power Viterbi–Viterbi estimator.
