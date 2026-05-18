"""High-level user-facing class for PyGaborSTM.

Wraps :class:`AuditorySpectrogram` and :class:`GaborFilterbank` behind a
single object so users don't have to manage the two stages by hand.
"""

import numpy as np

from .config import Config
from .spectrogram import AuditorySpectrogram
from .gabor import GaborFilterbank
from .structs import Spectrogram, RSF
from .backend import to_numpy


class PyGaborSTM:
    """Main interface for spectro-temporal modulation analysis.

    Holds an :class:`AuditorySpectrogram` and a :class:`GaborFilterbank`
    configured from the same :class:`Config`, and exposes both per-stage
    methods (:meth:`spectrogram`, :meth:`rsf`) and full-pipeline
    convenience methods (:meth:`compute`, :meth:`compute_device`).

    Parameters
    ----------
    config : Config, optional
        Configuration object. If ``None``, uses defaults.

    Attributes
    ----------
    config : Config
        The configuration used to build both internal stages.

    Examples
    --------
    >>> import pygaborstm as stm
    >>> model = stm.PyGaborSTM(config=stm.Config(use_gpu=True))
    >>> spec = model.spectrogram(audio)
    >>> rsf = model.rsf(spec)

    Or chained without an intermediate host transfer:

    >>> rsf = model.compute(audio)
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self._spec_model = AuditorySpectrogram(self.config)
        self._gabor_model = GaborFilterbank(self.config)

    def spectrogram(self, audio: np.ndarray) -> Spectrogram:
        """Compute the auditory spectrogram and return it on host.

        Parameters
        ----------
        audio : np.ndarray
            1D audio signal.

        Returns
        -------
        Spectrogram
            Spectrogram with data and axis metadata on the host.
        """
        return self._spec_model.compute(audio)

    def rsf(self, spec: Spectrogram) -> RSF:
        """Compute the RSF representation from a spectrogram, on host.

        Parameters
        ----------
        spec : Spectrogram
            Auditory spectrogram, typically produced by
            :meth:`spectrogram`.

        Returns
        -------
        RSF
            RSF representation with data and axis metadata on the host.
        """
        return self._gabor_model.compute(spec)

    # NOTE: chained compute_device runs ~15ms slower than calling
    # spectrogram()/rsf() separately on single files. Cause appears to be
    # device-side allocation pattern interacting with cuFFT — gabor stage runs
    # ~17ms when spec arrives via host-DMA upload but ~41ms when spec arrives
    # from a prior device kernel. Batch throughput is unaffected. Investigate
    # with nsys if it ever matters.
    def compute_device(self, audio: np.ndarray):
        """Run the full pipeline on device with no intermediate host transfer.

        Spectrogram output stays on the device and feeds directly into
        the Gabor stage. Use this when you do not need the intermediate
        :class:`Spectrogram` dataclass.

        Parameters
        ----------
        audio : np.ndarray
            1D audio signal.

        Returns
        -------
        np.ndarray or cupy.ndarray
            RSF tensor of shape
            ``(n_frames, n_rates, n_scales, n_freq)`` on the active
            backend.
        """
        spec_device = self._spec_model.compute_device(audio)
        return self._gabor_model.compute_device(spec_device)

    def compute(self, audio: np.ndarray) -> RSF:
        """Run the full pipeline and return an RSF dataclass on host.

        Chains both stages on device with no intermediate host transfer,
        then copies the final result to host and wraps it in an
        :class:`RSF`.

        Parameters
        ----------
        audio : np.ndarray
            1D audio signal.

        Returns
        -------
        RSF
            Host-side RSF with data and axis metadata.
        """
        rsf_device = self.compute_device(audio)
        rsf_data = to_numpy(rsf_device)

        frame_period = self.config.rsf_frame_shift_ms / 1000.0
        times = np.arange(rsf_data.shape[0]) * frame_period

        return RSF(
            data=rsf_data,
            times=times,
            rates=self._gabor_model.rates,
            scales=self._gabor_model.scales,
            freqs=self._spec_model.center_freqs,
        )
