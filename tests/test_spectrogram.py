import numpy as np

import pygaborstm as stm
from pygaborstm.spectrogram import AuditorySpectrogram
from pygaborstm.structs import Spectrogram


class TestAuditorySpectrogram:
    def test_output_type(self, audio_tone):
        model = stm.PyGaborSTM()
        result = model.spectrogram(audio_tone)
        assert isinstance(result, Spectrogram)

    def test_output_shape(self, audio_tone):
        model = stm.PyGaborSTM()
        result = model.spectrogram(audio_tone)
        assert result.n_freqs == 128  # default n_filters
        assert result.n_times > 0

    def test_no_nans(self, audio_tone):
        model = stm.PyGaborSTM()
        result = model.spectrogram(audio_tone)
        assert not np.any(np.isnan(result.data))

    def test_non_negative(self, audio_tone):
        """After rectification and compression, values should be non-negative."""
        model = stm.PyGaborSTM()
        result = model.spectrogram(audio_tone)
        assert np.all(result.data >= 0)

    def test_silence_near_zero(self, audio_silence):
        model = stm.PyGaborSTM()
        result = model.spectrogram(audio_silence)
        assert result.data.max() < 1e-6

    def test_tone_has_energy(self, audio_tone):
        model = stm.PyGaborSTM()
        result = model.spectrogram(audio_tone)
        assert result.data.max() > 0.01

    def test_noise_has_broadband_energy(self, audio_noise):
        """White noise should spread energy across channels."""
        model = stm.PyGaborSTM()
        result = model.spectrogram(audio_noise)
        channel_energy = result.data.mean(axis=1)
        active = np.sum(channel_energy > 1e-6)
        assert active > 0.8 * result.n_freqs

    def test_times_axis(self, audio_tone):
        model = stm.PyGaborSTM()
        result = model.spectrogram(audio_tone)
        assert len(result.times) == result.n_times
        assert result.times[0] == 0.0
        assert 0.9 < result.duration < 1.1

    def test_freqs_axis(self, audio_tone):
        model = stm.PyGaborSTM()
        result = model.spectrogram(audio_tone)
        assert len(result.freqs) == result.n_freqs
        assert result.freqs[0] < result.freqs[-1]  # Ascending

    def test_short_signal(self, audio_short):
        model = stm.PyGaborSTM()
        result = model.spectrogram(audio_short)
        assert result.n_times > 0

    def test_custom_n_filters(self, audio_tone):
        cfg = stm.Config(n_filters=64)
        model = stm.PyGaborSTM(config=cfg)
        result = model.spectrogram(audio_tone)
        assert result.n_freqs == 64

    def test_custom_sample_rate(self):
        sr = 8000
        t = np.linspace(0, 1, sr, endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t)
        model = stm.PyGaborSTM(config=stm.Config(sample_rate=sr))
        result = model.spectrogram(audio)
        assert result.sr == sr
        assert result.n_freqs == 128

    def test_2d_audio_flattened(self, audio_tone):
        """compute_device flattens >1D input."""
        model = AuditorySpectrogram()
        result = model.compute_device(audio_tone[None, :])  # (1, n_samples)
        assert result.ndim == 2  # (n_time, n_freq)


class TestGammatoneConfig:
    """Gammatone-specific config knobs flow into the SOS bank."""

    def test_filter_order_changes_sos_sections(self):
        default = AuditorySpectrogram()
        custom = AuditorySpectrogram(stm.Config(filter_order=8))
        # SOS bank shape is (n_filters, filter_order, 6).
        assert default._sos_device.shape[1] == 4
        assert custom._sos_device.shape[1] == 8

    def test_erb_scale_changes_filter_bandwidth(self, audio_tone):
        """Narrower ERB scale → narrower filters → more spectral concentration on a pure tone."""
        wide = AuditorySpectrogram(stm.Config(erb_scale=1.0))
        narrow = AuditorySpectrogram(stm.Config(erb_scale=0.3))

        wide_spec = wide.compute(audio_tone).data
        narrow_spec = narrow.compute(audio_tone).data

        # For a pure tone, narrower filters concentrate energy in fewer
        # channels. Count channels with above-threshold energy.
        threshold = 0.1 * wide_spec.max()
        wide_active = (wide_spec.mean(axis=1) > threshold).sum()
        narrow_active = (narrow_spec.mean(axis=1) > threshold).sum()
        assert narrow_active < wide_active


class TestLazyCache:
    """The y5 FFT cache is keyed on input length and rebuilt on length change."""

    def test_cache_built_on_first_call(self, audio_tone):
        model = AuditorySpectrogram()
        assert model._cached_n_samples is None
        model.compute_device(audio_tone)
        assert model._cached_n_samples == len(audio_tone)

    def test_cache_reused_for_same_length(self, audio_tone):
        model = AuditorySpectrogram()
        model.compute_device(audio_tone)
        kernel_fft_first = model._y5_kernel_fft
        model.compute_device(audio_tone)
        assert model._y5_kernel_fft is kernel_fft_first

    def test_cache_rebuilt_for_different_length(self, audio_tone, audio_short):
        model = AuditorySpectrogram()
        model.compute_device(audio_tone)
        kernel_fft_first = model._y5_kernel_fft
        model.compute_device(audio_short)
        assert model._y5_kernel_fft is not kernel_fft_first
        assert model._cached_n_samples == len(audio_short)

    def test_repeated_calls_same_result(self, audio_tone):
        """Cache reuse must not change output."""
        model = AuditorySpectrogram()
        out1 = model.compute_device(audio_tone)
        out2 = model.compute_device(audio_tone)
        np.testing.assert_array_equal(out1, out2)


class TestComputeDeviceOrientation:
    """compute_device returns (n_time, n_freq); compute() transposes to (n_freq, n_time)."""

    def test_orientations_match_after_transpose(self, audio_tone):
        model = AuditorySpectrogram()
        device_out = model.compute_device(audio_tone)  # (n_time, n_freq)
        host_spec = model.compute(audio_tone)  # (n_freq, n_time)
        np.testing.assert_array_equal(device_out.T, host_spec.data)
