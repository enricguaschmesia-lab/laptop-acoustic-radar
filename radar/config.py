"""Sounder configuration.

The preamble is identical to the EE-442 acoustic OFDM link (500 BPSK symbols at 1 kBd,
RRC roll-off 0.22, order-11 LFSR, carrier 8 kHz). The data symbols of the link are
replaced by a periodic OFDM sounding symbol: 64 subcarriers spaced 62.5 Hz between 6 and
10 kHz, so every 16 ms window of the recording yields one channel estimate.
"""

from dataclasses import asdict, dataclass

SPEED_OF_SOUND = 343.0  # m/s at ~20 degC


@dataclass(frozen=True)
class SounderConfig:
    fs: int = 48_000

    # Periodic OFDM sounding symbol
    n_sub: int = 64
    spacing: float = 62.5          # Hz -> period = fs / spacing = 768 samples = 16 ms
    f_low: float = 6_000.0         # first subcarrier (Hz)
    phase_scheme: str = "newman"   # "newman" (low crest factor) or "pn" (QPSK from the LFSR)

    # Single-carrier BPSK preamble, unchanged from the link
    f_c: float = 8_000.0
    sc_f_sym: int = 1_000
    sc_nsyms: int = 500
    rrc_rolloff: float = 0.22
    rrc_span_syms: int = 20        # one-sided RRC length in symbols

    # Frame layout and level
    lead_s: float = 0.5            # silence before the preamble (also the noise reference)
    tail_s: float = 0.5            # silence after the sounding train (catches latency)
    fade_s: float = 0.005          # raised-cosine fade of the sounding train
    amplitude: float = 0.5         # peak digital level of the output waveform

    @property
    def period(self) -> int:
        return round(self.fs / self.spacing)

    @property
    def k0(self) -> int:
        return round(self.f_low / self.spacing)

    @property
    def bandwidth(self) -> float:
        return self.n_sub * self.spacing

    @property
    def f_center(self) -> float:
        return self.f_low + self.bandwidth / 2

    @property
    def frame_rate(self) -> float:
        """Channel estimates per second (one per sounding period)."""
        return self.fs / self.period

    @property
    def tap_delay(self) -> float:
        """Delay resolution of the impulse response (s)."""
        return 1.0 / self.bandwidth

    @property
    def range_bin(self) -> float:
        """Monostatic range per impulse-response tap (m): c / (2B)."""
        return SPEED_OF_SOUND * self.tap_delay / 2

    @property
    def max_range(self) -> float:
        """Unambiguous monostatic range (m)."""
        return self.n_sub * self.range_bin

    @property
    def wavelength(self) -> float:
        return SPEED_OF_SOUND / self.f_center

    @property
    def max_speed(self) -> float:
        """Unambiguous radial speed (m/s) for Doppler measured across periods."""
        return (self.frame_rate / 2) * SPEED_OF_SOUND / (2 * self.f_center)

    @property
    def sc_os(self) -> int:
        return self.fs // self.sc_f_sym

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SounderConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
