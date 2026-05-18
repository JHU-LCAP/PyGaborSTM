import numpy as np

from pygaborstm.structs import Spectrogram, RSF


class TestSpectrogram:
    def setup_method(self):
        self.spec = Spectrogram(
            data=np.ones((128, 63)),
            times=np.arange(63) * 0.016,
            freqs=np.logspace(np.log2(180), np.log2(7000), 128, base=2),
            sr=16000,
        )

    def test_shape(self):
        assert self.spec.shape == (128, 63)

    def test_n_freqs(self):
        assert self.spec.n_freqs == 128

    def test_n_times(self):
        assert self.spec.n_times == 63

    def test_duration(self):
        assert self.spec.duration > 0

    def test_duration_single_frame(self):
        spec = Spectrogram(
            data=np.ones((128, 1)),
            times=np.array([0.0]),
            freqs=self.spec.freqs,
            sr=16000,
        )
        assert spec.duration == 0.0

    def test_to_numpy(self):
        arr = self.spec.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (128, 63)


class TestRSF:
    def setup_method(self):
        self.rsf = RSF(
            data=np.random.default_rng(42).random((33, 10, 6, 128)),
            times=np.arange(33) * 0.01,
            rates=np.array([-32, -16, -8, -4, -2, 2, 4, 8, 16, 32], dtype=float),
            scales=np.array([0.25, 0.5, 1, 2, 4, 8]),
            freqs=np.logspace(np.log2(180), np.log2(7000), 128, base=2),
        )

    def test_shape(self):
        assert self.rsf.shape == (33, 10, 6, 128)

    def test_n_frames(self):
        assert self.rsf.n_frames == 33

    def test_n_rates(self):
        assert self.rsf.n_rates == 10

    def test_n_scales(self):
        assert self.rsf.n_scales == 6

    def test_n_freqs(self):
        assert self.rsf.n_freqs == 128

    def test_to_numpy(self):
        arr = self.rsf.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (33, 10, 6, 128)

    def test_mean_over_time(self):
        result = self.rsf.mean_over_time()
        assert result.shape == (10, 6, 128)

    def test_mean_over_freq(self):
        result = self.rsf.mean_over_freq()
        assert result.shape == (33, 10, 6)

    def test_rate_scale_matrix(self):
        result = self.rsf.rate_scale_matrix()
        assert result.shape == (6, 10)  # [n_scales × n_rates]

    def test_rate_scale_matrix_folded(self):
        result = self.rsf.rate_scale_matrix(fold=True)
        assert result.shape == (6, 10)  # Same shape, symmetric

    def test_fold_is_symmetric(self):
        folded = self.rsf.rate_scale_matrix(fold=True)
        n_rates = folded.shape[1]
        mid = n_rates // 2
        left = np.flip(folded[:, :mid], axis=1)
        right = folded[:, mid:]
        np.testing.assert_array_almost_equal(left, right)

    def test_rate_scale_matrix_split(self):
        up, down = self.rsf.rate_scale_matrix_split()
        assert up.shape == (6, 5)
        assert down.shape == (6, 5)

    def test_upward_rates(self):
        up = self.rsf.upward_rates()
        np.testing.assert_array_equal(up, [-32, -16, -8, -4, -2])

    def test_downward_rates(self):
        down = self.rsf.downward_rates()
        np.testing.assert_array_equal(down, [2, 4, 8, 16, 32])

    def test_split_partitions_data(self):
        """Splitting + concatenating along rates axis should reconstruct original."""
        up, down = self.rsf._split_by_direction()
        recon = np.concatenate([up, down], axis=1)
        np.testing.assert_array_equal(recon, self.rsf.data)
