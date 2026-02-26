import numpy as np
import pytest

import pygaborstm as stm


@pytest.fixture
def sr():
    """Default sample rate."""
    return 16000


@pytest.fixture
def audio_tone(sr):
    """1-second 440 Hz sine wave."""
    t = np.linspace(0, 1, sr, endpoint=False)
    return np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def audio_silence(sr):
    """1 second of silence."""
    return np.zeros(sr)


@pytest.fixture
def audio_noise(sr):
    """1 second of white noise."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(sr)


@pytest.fixture
def audio_short(sr):
    """Very short audio (100 ms)."""
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    return np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def spectrogram_from_tone(audio_tone):
    """Pre-computed spectrogram from 440 Hz tone."""
    model = stm.PyGaborSTM()
    return model.spectrogram(audio_tone)
