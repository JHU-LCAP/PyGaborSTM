"""Configuration dataclass for PyGaborSTM.

Default values match Bellur & Elhilali (2017).
"""

import warnings
from dataclasses import dataclass, field, fields

import numpy as np

from .constants import RESOLUTION_MULTIPLIERS, STANDARD_RATES, STANDARD_SCALES


@dataclass(eq=False)
class Config:
    """Configuration for the full PyGaborSTM pipeline.

    Parameters
    ----------
    use_gpu : bool, default False
        If True, use CuPy for GPU acceleration. Silently falls back to
        NumPy with a warning if CuPy is unavailable.
    sample_rate : int, default 16000
        Audio sample rate in Hz. Must match the input signal.
    n_filters : int, default 128
        Number of cochlear (gammatone) channels in the spectrogram stage.
    f_min : float, default 180.0
        Lowest filter center frequency in Hz.
    octaves : float, default 5.3
        Total frequency range in octaves. ``f_max = f_min * 2**octaves``.
    tau_ms : float, default 8.0
        Time constant (ms) for the leaky-integration stage of the
        spectrogram.
    frmlen_ms : float, default 16.0
        Spectrogram frame length in ms (downsampling factor after y5).
    filter_order : int, default 4
        Number of biquad cascades per gammatone filter. Higher values
        produce sharper filters at proportionally higher cost.
    erb_scale : float, default 0.6
        Multiplier on the Glasberg-Moore ERB bandwidth. Values below
        1.0 narrow the filters; 0.6 matches the NSL toolbox default.
    rates : np.ndarray, default ``STANDARD_RATES``
        Base set of temporal modulation rates (Hz). Used as anchor points;
        ``resolution`` may interpolate additional rates between them.
    scales : np.ndarray, default ``STANDARD_SCALES``
        Base set of spectral modulation scales (cycles/octave). Anchor
        points for ``resolution`` interpolation.
    resolution : {"low", "medium", "high", "ultra", "max", "overkill"}, default "low"
        Density multiplier for the rate and scale grids. Each level
        doubles the number of kernels along both axes.
    rsf_frame_size_ms : int, default 500
        RSF integration window length in ms.
    rsf_frame_shift_ms : int, default 10
        RSF hop size in ms.
    """

    # General
    use_gpu: bool = False
    sample_rate: int = 16000

    # Spectrogram
    n_filters: int = 128
    f_min: float = 180.0
    octaves: float = 5.3
    tau_ms: float = 8.0
    frmlen_ms: float = 16.0

    # Gammatone
    filter_order: int = 4
    erb_scale: float = 0.6

    # RSF / Gabor
    rates: np.ndarray = field(default_factory=lambda: STANDARD_RATES.copy())
    scales: np.ndarray = field(default_factory=lambda: STANDARD_SCALES.copy())
    resolution: str = "low"
    rsf_frame_size_ms: int = 500
    rsf_frame_shift_ms: int = 10

    def __post_init__(self) -> None:
        self.rates = np.asarray(self.rates, dtype=np.float64)
        self.scales = np.asarray(self.scales, dtype=np.float64)
        self.validate()
        self._warn_above_nyquist()

    def _warn_above_nyquist(self) -> None:
        # Advisory only, and only at construction: the pipeline stages
        # re-run validate() and should not repeat this.
        f_max = self.f_min * (2.0**self.octaves)
        if f_max > self.sample_rate / 2:
            warnings.warn(
                f"Highest cochlear filter ({f_max:.0f} Hz) is above Nyquist "
                f"({self.sample_rate / 2:.0f} Hz); the top channels will alias. "
                f"Lower octaves or raise sample_rate.",
                UserWarning,
                stacklevel=3,
            )

    def validate(self) -> None:
        """Reject parameter combinations that cannot produce valid output.

        Run at construction and again by each pipeline stage, so a config
        mutated after construction is still caught before it reaches a
        cryptic SciPy error or a silently all-NaN result.

        Raises
        ------
        ValueError
            If any parameter or combination is invalid.
        """
        positive = (
            ("sample_rate", self.sample_rate),
            ("n_filters", self.n_filters),
            ("f_min", self.f_min),
            ("octaves", self.octaves),
            ("tau_ms", self.tau_ms),
            ("frmlen_ms", self.frmlen_ms),
            ("filter_order", self.filter_order),
            ("erb_scale", self.erb_scale),
            ("rsf_frame_size_ms", self.rsf_frame_size_ms),
            ("rsf_frame_shift_ms", self.rsf_frame_shift_ms),
        )
        for name, value in positive:
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}.")

        samples_per_frame = int((self.frmlen_ms / 1000.0) * self.sample_rate)
        if samples_per_frame < 1:
            raise ValueError(
                f"frmlen_ms={self.frmlen_ms} at sample_rate={self.sample_rate} Hz "
                f"gives {samples_per_frame} samples per frame; at least 1 is "
                f"required. Use frmlen_ms >= {1000.0 / self.sample_rate:g}."
            )

        # A window shorter than one spectrogram frame integrates zero samples
        # and every RSF value comes back NaN, with no error.
        if int(self.rsf_frame_size_ms / self.frmlen_ms) < 1:
            raise ValueError(
                f"rsf_frame_size_ms={self.rsf_frame_size_ms} is shorter than one "
                f"spectrogram frame (frmlen_ms={self.frmlen_ms}), so the RSF "
                f"window would be empty and the output entirely NaN. "
                f"Use rsf_frame_size_ms >= frmlen_ms."
            )

        if self.resolution not in RESOLUTION_MULTIPLIERS:
            raise ValueError(
                f"Invalid resolution '{self.resolution}'. "
                f"Choose from {list(RESOLUTION_MULTIPLIERS)}"
            )

        rates = np.asarray(self.rates, dtype=np.float64)
        scales = np.asarray(self.scales, dtype=np.float64)
        if rates.size == 0 or scales.size == 0:
            raise ValueError("rates and scales must each contain at least one value.")
        if not np.any(rates > 0):
            raise ValueError("rates must contain at least one positive value.")
        if np.any(scales <= 0):
            raise ValueError("scales must all be > 0 cycles/octave.")

    def __eq__(self, other: object) -> bool:
        # The generated __eq__ compares the ndarray fields and raises
        # "truth value of an array is ambiguous".
        if other.__class__ is not self.__class__:
            return NotImplemented
        for f in fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                if not np.array_equal(a, b):
                    return False
            elif a != b:
                return False
        return True
