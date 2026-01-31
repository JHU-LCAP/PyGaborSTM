import numpy as np
import pytest

import pygaborstm as stm
from pygaborstm.config import Config


SR = 16000


@pytest.fixture
def sr():
    return SR


@pytest.fixture
def default_config():
    return Config()


@pytest.fixture
def audio_silence(sr):
    """1 second of silence."""
    return np.zeros(sr)


@pytest.fixture
def audio_tone(sr):
    """1 second of 440 Hz sine tone."""
    t = np.linspace(0, 1, sr, endpoint=False)
    return 0.5 * np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def audio_noise(sr):
    """1 second of white noise."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(sr)


@pytest.fixture
def audio_short(sr):
    """100ms of tone (short signal edge case)."""
    n = int(0.1 * sr)
    t = np.linspace(0, 0.1, n, endpoint=False)
    return 0.5 * np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def spectrogram_from_tone(audio_tone):
    """Pre-computed spectrogram from 440 Hz tone."""
    return stm.auditory_spectrogram(audio_tone)


@pytest.fixture
def rsf_from_tone(spectrogram_from_tone):
    """Pre-computed RSF from 440 Hz tone."""
    model = stm.GaborFilterbank()
    return model.compute(spectrogram_from_tone)
