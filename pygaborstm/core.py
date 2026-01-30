from pathlib import Path

import numpy as np
import soundfile as sf

from .config import Config, SpectrogramConfig, GaborConfig
from .spectrogram import AuditorySpectrogram
from .gabor import GaborFilterbank
from .structs import Spectrogram, RSF


def load(path: str | Path) -> tuple[np.ndarray, int]:
    """
    Load audio file.

    Args:
        path: Path to audio file

    Returns:
        Tuple of (audio array, sample rate)
    """
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio, sr


def compute_rsf(
    source: str | Path | np.ndarray,
    sr: int | None = None,
    config: Config | None = None,
    **kwargs,
) -> RSF:
    """
    Compute RSF representation from audio.

    Args:
        source: Audio file path or numpy array
        sr: Sample rate (required if source is array)
        config: Configuration object (optional)
        **kwargs: Override individual config parameters

    Returns:
        RSF object
    """
    if isinstance(source, (str, Path)):
        audio, sr = load(source)
    else:
        audio = source
        if sr is None:
            raise ValueError("Sample rate (sr) required when source is array")

    cfg = config or Config()
    cfg = _apply_overrides(cfg, kwargs)

    if sr != cfg.spectrogram.sample_rate:
        audio = _resample(audio, sr, cfg.spectrogram.sample_rate)

    spec_model = AuditorySpectrogram(cfg.spectrogram)
    spectrogram = spec_model.compute(audio)

    gabor_model = GaborFilterbank(cfg.gabor, cfg.spectrogram)
    rsf = gabor_model.compute(spectrogram)

    return rsf


def _apply_overrides(cfg: Config, overrides: dict) -> Config:
    """Apply keyword overrides to config."""
    if not overrides:
        return cfg

    spectrogram_keys = {
        "sample_rate", "n_filters", "f_min", "octaves",
        "tau_ms", "frmlen_ms", "constant_Q",
    }

    gabor_keys = {
        "n_freq_bins", "resolution", "rsf_frame_size_ms", "rsf_frame_shift_ms",
    }

    for key, value in overrides.items():
        if key == "use_gpu":
            cfg.use_gpu = value
        elif key in spectrogram_keys:
            setattr(cfg.spectrogram, key, value)
        elif key in gabor_keys:
            setattr(cfg.gabor, key, value)
        else:
            raise ValueError(f"Unknown parameter: {key}")

    return cfg


def _resample(audio: np.ndarray, sr_orig: int, sr_target: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    import librosa
    return librosa.resample(audio, orig_sr=sr_orig, target_sr=sr_target)