import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import pygaborstm as stm
from pygaborstm.structs import RSF


@pytest.fixture
def wav_mono(audio_tone, sr):
    """Write a mono wav file and return its path."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio_tone, sr)
        return Path(f.name)


@pytest.fixture
def wav_stereo(audio_tone, sr):
    """Write a stereo wav file and return its path."""
    stereo = np.column_stack([audio_tone, audio_tone * 0.5])
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, stereo, sr)
        return Path(f.name)


@pytest.fixture
def wav_8k(sr):
    """Write an 8kHz wav file for resampling test."""
    sr_8k = 8000
    t = np.linspace(0, 1, sr_8k, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sr_8k)
        return Path(f.name)


class TestLoad:
    def test_load_mono(self, wav_mono):
        audio, sr = stm.load(wav_mono)
        assert audio.ndim == 1
        assert sr == 16000
        assert len(audio) > 0

    def test_load_stereo_downmixes(self, wav_stereo):
        audio, sr = stm.load(wav_stereo)
        assert audio.ndim == 1

    def test_load_returns_path_types(self, wav_mono):
        # str path
        audio1, _ = stm.load(str(wav_mono))
        # Path object
        audio2, _ = stm.load(wav_mono)
        np.testing.assert_array_equal(audio1, audio2)


class TestComputeRSF:
    def test_from_array(self, audio_tone, sr):
        result = stm.compute_rsf(audio_tone, sr=sr)
        assert isinstance(result, RSF)
        assert result.n_rates == 10
        assert result.n_scales == 6
        assert result.n_freqs == 256

    def test_missing_sr_raises(self, audio_tone):
        with pytest.raises(ValueError, match="Sample rate"):
            stm.compute_rsf(audio_tone)

    def test_kwargs_override(self, audio_tone, sr):
        result = stm.compute_rsf(audio_tone, sr=sr, resolution="low")
        assert isinstance(result, RSF)

    def test_invalid_kwarg_raises(self, audio_tone, sr):
        with pytest.raises(ValueError, match="Unknown parameter"):
            stm.compute_rsf(audio_tone, sr=sr, fake_param=42)

    def test_config_object(self, audio_tone, sr):
        cfg = stm.Config(
            spectrogram=stm.SpectrogramConfig(n_filters=64),
        )
        result = stm.compute_rsf(audio_tone, sr=sr, config=cfg)
        assert result.n_freqs == 64

    def test_from_file_path(self, wav_mono):
        result = stm.compute_rsf(wav_mono)
        assert isinstance(result, RSF)
        assert result.n_freqs == 256

    def test_from_file_str(self, wav_mono):
        result = stm.compute_rsf(str(wav_mono))
        assert isinstance(result, RSF)

    def test_resamples_mismatched_sr(self, wav_8k):
        """8kHz file should be resampled to 16kHz internally."""
        result = stm.compute_rsf(wav_8k)
        assert isinstance(result, RSF)
        assert result.n_freqs == 256

    def test_deterministic(self, audio_tone, sr):
        """Same input should produce same output."""
        r1 = stm.compute_rsf(audio_tone, sr=sr)
        r2 = stm.compute_rsf(audio_tone, sr=sr)
        np.testing.assert_array_equal(r1.data, r2.data)


class TestApplyOverrides:
    def test_use_gpu_override(self, audio_tone, sr):
        result = stm.compute_rsf(audio_tone, sr=sr, use_gpu=False)
        assert isinstance(result, RSF)

    def test_spectrogram_key_override(self, audio_tone, sr):
        result = stm.compute_rsf(audio_tone, sr=sr, n_filters=64)
        assert result.n_freqs == 64

    def test_gabor_key_override(self, audio_tone, sr):
        result = stm.compute_rsf(audio_tone, sr=sr, rsf_frame_size_ms=250)
        assert isinstance(result, RSF)


class TestEndToEnd:
    def test_full_pipeline_shapes(self, audio_tone, sr):
        """Verify shapes are consistent through the pipeline."""
        spec = stm.auditory_spectrogram(audio_tone)
        assert spec.n_freqs == 256

        model = stm.GaborFilterbank()
        rsf = model.compute(spec)
        assert rsf.n_freqs == spec.n_freqs
        assert rsf.n_rates == len(model.rates)
        assert rsf.n_scales == len(model.scales)

    def test_convenience_matches_manual(self, audio_tone, sr):
        """compute_rsf should produce same result as manual pipeline."""
        # Convenience
        rsf_auto = stm.compute_rsf(audio_tone, sr=sr)

        # Manual
        spec = stm.auditory_spectrogram(audio_tone)
        model = stm.GaborFilterbank()
        rsf_manual = model.compute(spec)

        np.testing.assert_array_almost_equal(rsf_auto.data, rsf_manual.data, decimal=10)
