import numpy as np

import pygaborstm as stm
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
        """Silence should produce near-zero output."""
        model = stm.PyGaborSTM()
        result = model.spectrogram(audio_silence)
        assert result.data.max() < 1e-6

    def test_tone_has_energy(self, audio_tone):
        """A tone should produce non-trivial energy."""
        model = stm.PyGaborSTM()
        result = model.spectrogram(audio_tone)
        assert result.data.max() > 0.01

    def test_noise_has_broadband_energy(self, audio_noise):
        """White noise should spread energy across channels."""
        model = stm.PyGaborSTM()
        result = model.spectrogram(audio_noise)
        channel_energy = result.data.mean(axis=1)
        # At least 80% of channels should have nonzero energy
        active = np.sum(channel_energy > 1e-6)
        assert active > 0.8 * result.n_freqs

    def test_times_axis(self, audio_tone):
        model = stm.PyGaborSTM()
        result = model.spectrogram(audio_tone)
        assert len(result.times) == result.n_times
        assert result.times[0] == 0.0
        # Duration should be close to 1 second
        assert 0.9 < result.duration < 1.1

    def test_freqs_axis(self, audio_tone):
        model = stm.PyGaborSTM()
        result = model.spectrogram(audio_tone)
        assert len(result.freqs) == result.n_freqs
        assert result.freqs[0] < result.freqs[-1]  # Ascending

    def test_short_signal(self, audio_short):
        """Should handle short signals without crashing."""
        model = stm.PyGaborSTM()
        result = model.spectrogram(audio_short)
        assert result.n_times > 0

    def test_custom_config(self, audio_tone):
        cfg = stm.Config(n_filters=64)
        model = stm.PyGaborSTM(config=cfg)
        result = model.spectrogram(audio_tone)
        assert result.n_freqs == 64
