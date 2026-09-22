"""High-level user-facing class for PyGaborSTM.

Wraps :class:`AuditorySpectrogram` and :class:`GaborFilterbank` behind a
single object so users don't have to manage the two stages by hand.
"""

import numpy as np

from .backend import to_numpy
from .config import Config
from .gabor import GaborFilterbank
from .spectrogram import AuditorySpectrogram
from .structs import RSF, Spectrogram


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

    @property
    def device(self):
        """The backend actually in use.

        ``config.use_gpu`` is the request; this is the answer. Check
        ``model.device.on_gpu`` to confirm a GPU request took effect.
        """
        return self._spec_model.device

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

        return RSF(
            data=rsf_data,
            times=self._gabor_model.frame_times(rsf_data.shape[0]),
            rates=self._gabor_model.rates,
            scales=self._gabor_model.scales,
            freqs=self._spec_model.center_freqs,
        )
