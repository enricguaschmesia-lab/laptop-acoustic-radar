# Lab notebook

How the radar developed over two days (19–20 October 2026), kept close to how it was written at
the time. The [README](../README.md) has the final results; this page keeps the intermediate
findings, the dead ends and the per-session numbers.

The sounder: the EE-442 acoustic OFDM link turned into a monostatic sonar on the laptop's own
speaker and microphones (8 kHz carrier, 4.3 cm wavelength, about the same as UWB radar at 7–8 GHz).

| | Value |
|---|---|
| Preamble | unchanged from the link: 500 BPSK symbols at 1 kBd, RRC 0.22, LFSR x^11+x^2+1, 8 kHz carrier |
| Sounding | periodic OFDM symbol, 64 subcarriers x 62.5 Hz from 6 to 10 kHz, Newman phases (crest factor < 6 dB) |
| Channel estimate | LS per subcarrier every 16 ms (62.5 Hz), then IFFT to a 64-tap impulse response |
| Range bin / max range | 4.3 cm / 2.7 m |
| Max unambiguous speed | 0.67 m/s |

## First hardware capture (2026-10-19)
- Port verified against MATLAB (PN, RRC, preamble and frame-sync metric identical); 14 tests pass.
- First hardware capture (WASAPI shared mode, speakers at amplitude 0.5): in-band SNR 43-44 dB,
  latency 136 ms, speaker and mic on one clock (phase drift < 1 deg/s). The scene was not static
  (user at the keyboard, speakers muted at 14.4 s), so the stability checks are inconclusive. In the
  quiet stretch, mic channel 1 passed (amplitude std 3.7 %, phase jitter 2.9 deg); channel 0's amplitude
  sagged by 42 % with a stable phase. The second capture (WASAPI exclusive) was taken with the speakers
  muted and is unusable.

## Go/no-go session 20261020-173035: GO (mic channel 1 passes all checks)
| Check | Mic 0 | Mic 1 |
|---|---|---|
| Empty room (static) | SNR 81 dB, amp std 0.9 %, phase jitter 0.4 deg, floor -51 dB | SNR 80 dB, amp std 1.4 %, phase jitter 0.4 deg, floor -48.5 dB |
| Book at 30 / 60 cm (vs empty) | 47 cm (13.7 dB) / 60 cm (8.0 dB): difference 12.9 cm, **FAIL** (true 30 cm peak 2 dB lower) | 34 cm (9.0 dB) / 60 cm (13.4 dB): difference 25.7 cm, PASS |
| Book pulled 30 -> 70 cm | 34 -> 69 cm, PASS | 21 -> 69 cm (flicker once the book stops), PASS |
| Paced breathing, cue 15/min | 15.6/min at the 60 cm bin, PASS | 15.4/min at the 99 cm bin, PASS |

Lessons for the dataset: moving targets (pull, breathing) are robust, while static detection by
subtracting captures is fragile (other peaks within 3-5 dB of the book), so features should come
from the clutter-removed response. The breathing bins sit further out than the 40 cm chest mark
(slant range to a seated torso, or multipath), which is still to be understood.

## Dataset (`scripts/collect.py`, protocol in `radar/protocol.py`)
Four sessions of 7 steps (6 min of sound, about 10 min each): empty room (60 s), sitting still (60 s,
at 40/70/100 cm by session), paced breathing at three chair distances (3 x 40 s; rates 10/15/20 per
minute in a Latin square over sessions 1-3, session 4 repeats session 1), walking between 0.8 and
2.5 m (60 s), hand waving at 30-60 cm while seated at 70 cm (60 s). Classes: empty, still,
walking, waving. Captures go to `data/raw/dataset/session_<n>/` with a `manifest.json`; each
capture gets a quick check (sync, level, motion energy, breathing rate against the cue) so a bad
take can be redone on the spot, and an interrupted session can be resumed.
```
python scripts/collect.py --session 1 --simulate   # dry run, data/raw/dataset_sim
python scripts/collect.py --session 1              # plays sound, Enter per step
python scripts/collect.py --session 1 --check      # re-check offline
```

### Session 1 (2026-10-20): complete
| Step | Motion level ch0/ch1 (dB re total) | Breathing estimate (cue) |
|---|---|---|
| empty | -46.5 / -45.3 | - |
| still, 40 cm | -24.2 / -20.5 | 12.9/min natural |
| paced, 40 cm | -20.6 / -18.2 | 10.2 / 10.1 (10) |
| paced, 70 cm | -25.7 / -23.5 | 15.1 / 15.0 (15) |
| paced, 100 cm | -24.5 / -23.2 | 20.1 / 20.1 (20) |
| walking 0.8-2.5 m | -17.8 / -17.7 | - |
| waving 30-60 cm | -20.2 / -19.9 | - |

Findings:
- Deep breaths at 40-70 cm move the chest 2-3 cm, more than half the 4.3 cm wavelength: the I/Q
  points wrap around the circle and a straight-line projection doubles the rate (20/min for a
  10/min cue, 30 for 15). The estimator now votes across the top 4 range bins and two
  demodulators (Kasa circle-fit arc phase, principal-axis projection), drops spectrally impure
  candidates, and resolves the f/2f ambiguity towards f. Same failure as phase wrapping in
  60 GHz mmWave vital-sign radar (5 mm wavelength).
- Presence is easy (every occupied step is 20-29 dB above the empty room); activities overlap
  in energy (still -20 to -26 dB, waving -20, walking -18), so separating them needs
  range/Doppler structure: the case for the CNN.
- Breathing bins sit further out than the chest (47 cm for a 40 cm chest, 124-159 cm at 100 cm).

### Session 2 (2026-10-20): complete, one step to redo
| Step | Motion level ch0/ch1 (dB re total) | Breathing estimate (cue) |
|---|---|---|
| empty | -45.4 / -46.0 | - |
| still, 70 cm | -23.2 / -23.1 | 15.1/min natural |
| paced, 40 cm | -17.1 / -17.4 | 15.1 / 15.1 (15) |
| paced, 70 cm | -22.3 / -21.8 | 20.4 / 20.4 (20) |
| paced, 100 cm (redo) | -27.4 / -27.0 | 9.8 / 10.0 (10); the first take (operator at ~1.2 m) is superseded and noted in the manifest |
| walking 0.8-2.5 m | -18.0 / -17.9 | - |
| waving 30-60 cm | -21.3 / -21.3 | - |

Breathing bins matched the chest this time at 70 cm (64-73 cm), and mic 1 put the misplaced
100 cm capture at 120 cm, where the operator actually sat.

### Breathing estimator frozen (2026-10-20)
Tuned on sessions 1-2 plus the go/no-go capture: arc demodulation, multi-bin voting, f/2f
resolution, and the rate taken as the winning group's median (median vs purest candidate on the 12
tuning estimates: mean absolute error 0.157 vs 0.211/min, worst 0.39 vs 0.83). On the tuning data,
every paced estimate is within 0.4/min of its cue. Sessions 3-4 are the held-out test and must be
scored with this exact file: `radar/processing.py` sha256
`0473e78c290b4dd4e1f9c4361f39da03001aa3d1ed3ad007a19ff30981e57962`.

### Session 3 (2026-10-20): complete, held-out breathing test
| Step | Motion level ch0/ch1 (dB re total) | Breathing estimate (cue) |
|---|---|---|
| empty | -45.1 / -45.7 | - |
| still, 100 cm | -28.2 / -27.6 | 14.0/min natural |
| paced, 40 cm | -17.2 / -17.7 | 20.47 / 20.44 (20) |
| paced, 70 cm | -25.9 / -26.1 | 10.31 / 10.25 (10) |
| paced, 100 cm | -25.0 / -23.4 | 15.16 / 15.12 (15) |
| walking 0.8-2.5 m | -17.7 / -17.6 | - |
| waving 30-60 cm | -19.2 / -18.8 | - |

| Breathing, frozen estimator | n | MAE | max | mean bias |
|---|---|---|---|---|
| tuning (sessions 1-2) | 12 | 0.154/min | 0.39 | +0.106 |
| held-out (session 3) | 6 | 0.292/min | 0.47 | +0.292 |

Every estimate (18/18) is above its cue. Open question: the operator leading the tone, or an
estimator bias. The synthetic tests show no upward bias (9.9/15.0/20.0), and the playback/recording
clock cancels out, so leading the cue is the main suspect. To test: compare the breathing
signal's phase drift against the cue's phase over each capture.

### Session 4 (2026-10-20) and the complete dataset
Session 4 (repeat of session 1) was clean: paced breathing 10.04/10.05 (cue 10, the condition that
doubled in session 1), 15.01/15.06 (15), 20.09/20.14 (20).

Dataset: 4 sessions, 28 captures in use (1 superseded), 24 min of sound: empty 4 min, still 12 min
(sitting + paced breathing), walking 4 min, waving 4 min. No capture has a warning.

The cue is what the operator *tried* to follow, not exact ground truth, so the deviations below
combine the person's pacing and the tracker's error. They are an upper bound on tracker error
(see "Breathing check" below: the tone itself measures exact, and the remaining offset is the
person running slightly ahead of it).

| Breathing deviation from the cue, frozen estimator (sha256 0473e78c...) | n | mean abs | RMS | max | mean | within 0.5/min |
|---|---|---|---|---|---|---|
| tuning (sessions 1-2) | 12 | 0.154/min | 0.193 | 0.39 | +0.106 | 100 % |
| **held-out (sessions 3-4)** | 12 | **0.178/min** | 0.233 | 0.47 | +0.178 | 100 % |
| held-out session 3 | 6 | 0.292 | 0.320 | 0.47 | +0.292 | 100 % |
| held-out session 4 | 6 | 0.064 | 0.078 | 0.14 | +0.064 | 100 % |

Motion level (dB re total, mean of both mics) per class and session:

| | s1 | s2 | s3 | s4 |
|---|---|---|---|---|
| empty | -45.9 | -45.7 | -45.4 | -45.9 |
| still | -22.6 | -22.4 | -23.9 | -21.8 |
| walking | -17.7 | -18.0 | -17.6 | -17.7 |
| waving | -20.0 | -21.3 | -19.0 | -20.1 |

### Breathing check (`scripts/breathing_check.py`, `results/breathing/`)
Is the small positive deviation from the cue the tracker or the person?
- **Tracker and time base verified:** the pacing tone is audible in every paced recording. Its pitch,
  tracked once per sounding period and run through the same frozen spectral estimator on the same
  frames, reads 10.000 / 15.000 / 20.000 per minute (24 mic-captures, max error 0.0002/min).
- **The person runs slightly ahead of the tone:** the breathing-minus-tone phase drift, which does
  not use the spectral peak at all, gives a mean offset of +0.136/min. The radar estimate minus the
  cue gives +0.142/min (correlation 0.64 across captures, mean abs difference 0.13/min).
- So most of the deviation from the cue is human pacing, not tracker error. A few drift estimates
  are noisy (skipped or extra breaths, phase slips; e.g. session 2's 100 cm redo on mic 0), so this
  is strong evidence, not proof. Without an independent reference such as a chest belt, tracker error
  against true breathing cannot be measured directly.

## Classifiers (`scripts/build_windows.py`, `scripts/train_eval.py`, `results/ml/`)
Inputs: 4.1 s windows (256 periods) every 1 s, both mics; static clutter removed per window
(window mean), powers normalised by the window's static power. 1300 windows: empty 220,
still 640, walking 220, waving 220. Evaluation: leave-one-session-out (train on three sessions,
test on the fourth, all four rotations); overlapping windows never cross that boundary.

| Model | Accuracy | Macro-F1 | Recall empty/still/walking/waving |
|---|---|---|---|
| Majority class | 0.492 | 0.165 | 0/1/0/0 |
| Energy only, logistic regression | 0.541 | 0.556 | 1.00/0.42/0.76/0.22 |
| 10 hand-crafted features + logistic regression | 0.996 | 0.995 | 1.00/1.00/1.00/0.98 |
| 10 hand-crafted features + random forest | 0.998 | 0.997 | 1.00/1.00/1.00/0.99 |
| Small CNN on range-Doppler + range-time maps (3 seeds) | 0.999 | 0.998 | 1.00/1.00/1.00/0.99 |

- Presence (empty vs occupied): one energy threshold (learned per fold at -29.4 to -30.6 dB) is perfect.
- Activities: energy alone fails (still is confused with walking and waving), but range and Doppler
  structure separates them almost perfectly. The top random-forest features are the slow-motion
  fraction (|f_d| < 1 Hz), the range extent of motion, motion energy and the fast-motion fraction.
  The position features (peak range) rank near the bottom, so the model does not just key on where
  the person stood. "Still" without the pacing tone is recognised as well as with it (no cue leakage).
- CNN: 24k parameters, 1.8 ms per window on one CPU thread, 45 s training per fold; fixed 30-epoch
  schedule (no early stopping, so the test session never influences training), class-weighted loss,
  augmentation by mic swap, Doppler/time mirroring, ±2 range-bin shifts, level offset and noise.
- **What not to trust:** this is a ceiling. One person, one room, four scripted activities that differ
  grossly in range and speed, so cross-session tests are easy and the benchmark cannot separate the
  CNN from the hand-crafted features. Nothing here tests a new room, a new person, several people, or
  harder classes (e.g. typing vs reading). Pooled metrics count overlapping windows, so per-session
  numbers (macro-F1 0.99-1.00 for features/CNN, 0.39-0.67 for energy) are the honest spread.

## Stress test: noise and window length (`scripts/stress_test.py`, `results/stress/`)
Complex white noise added to the measured impulse responses before windowing, in dB relative to
each capture's total received power (the same scale as the motion levels: still -22.7, waving
-20.1, walking -17.8 dB). Train and test get the same noise (matched); leave-one-session-out; one
CNN seed per condition (30 epochs). Mean macro-F1 over the four test sessions:

| Added noise | Energy 4 s / 1 s | Features+LR 4 s / 1 s | Features+RF 4 s / 1 s | CNN 4 s / 1 s | Presence: energy, CNN (4 s) |
|---|---|---|---|---|---|
| none | 0.55 / 0.72 | 0.995 / 0.99 | 0.997 / 0.99 | 0.999 / 0.997 | 1.00, 1.00 |
| -30 dB | 0.54 / 0.70 | 0.995 / 0.99 | 1.000 / 0.99 | 1.000 / 0.994 | 1.00, 1.00 |
| -20 dB | 0.54 / 0.67 | 0.965 / 0.97 | 0.982 / 0.99 | **1.000** / 0.973 | 1.00, 1.00 |
| -15 dB | 0.49 / 0.63 | 0.965 / 0.95 | 0.978 / 0.98 | **0.999** / 0.942 | 1.00, 1.00 |
| -10 dB | 0.46 / 0.54 | 0.952 / 0.90 | 0.966 / **0.92** | **0.982** / 0.829 | 0.99, 1.00 |
| -5 dB | 0.44 / 0.40 | **0.913** / 0.68 | 0.908 / 0.68 | 0.685 / 0.550 | 0.92, 0.92 |
| 0 dB | 0.36 / 0.29 | **0.767** / 0.44 | 0.724 / 0.42 | 0.387 / 0.295 | 0.80, 0.65 |

Findings:
- **Processing gain:** with 4 s windows every structured model still scores >= 0.95 at -10 dB, noise 8 dB
  above walking's and 13 dB above sitting still's motion level. Coherent integration over 256 periods
  (Doppler FFT, ~24 dB) is doing the work; energy alone never gets past 0.55.
- **CNN vs features depends on the regime:** with 4 s windows the CNN is the most robust model up to -10 dB
  (still perfect at -15 dB, where the features have slipped to 0.97), then collapses (0.69 at -5 dB,
  0.39 at 0 dB) in all four test sessions, while the features degrade gracefully (0.91, 0.77). With
  1 s windows the random forest leads from -20 dB on. Plausible reason: the features hard-code
  averaging over the whole range-Doppler map (slow/fast fractions, mean speed), whereas a CNN trained
  from scratch on ~1000 windows has not learned that averaging and fits noise patterns. Presence shows
  the same: at 0 dB the plain energy threshold (0.80) beats the CNN (0.65).
- **Window length buys about 5 dB:** 4 s windows hold any given score about 5 dB deeper into the
  noise than 1 s windows, close to the 6 dB coherent-gain ratio of 4x longer integration.
- **What not to trust:** one CNN seed per condition (seed spread was ~0 at the ceiling, larger
  under heavy noise; the bands show only the spread over test sessions), synthetic white noise rather
  than real interference, and the same one-person, one-room data as above.
- **Next:** train the CNN with noise augmentation across levels, more seeds and more data; give it the
  hand-crafted features as extra inputs (hybrid); average power in the linear domain before the log.

Presence is separable by energy alone (empty sits 20+ dB below everything else in every session).
The activity classes are a few dB apart on average, but individual captures overlap (paced
breathing at 40 cm reaches -17 dB, the walking level), so separating activities needs range and
Doppler structure.
