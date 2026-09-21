"""Default rate and scale grids for Gabor filter tuning.

Values follow Bellur & Elhilali (2017) and the NSL Toolbox conventions:
negative rates select upward-sweeping ripples, positive rates select
downward, and scales are expressed in cycles per octave.
"""

import numpy as np

STANDARD_RATES = np.array([-32, -16, -8, -4, -2, 2, 4, 8, 16, 32], dtype=np.float64)
"""Temporal modulation rates (Hz). Symmetric about zero."""

STANDARD_SCALES = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=np.float64)
"""Spectral modulation scales (cycles/octave). Logarithmically spaced."""

RESOLUTION_MULTIPLIERS = {
    "low": 1,
    "medium": 2,
    "high": 4,
    "ultra": 8,
    "max": 16,
    "overkill": 32,
}
"""Rate/scale grid density per ``Config.resolution`` level.

Lives here rather than on :class:`~pygaborstm.gabor.GaborFilterbank` so
that :class:`~pygaborstm.config.Config` can validate against it without
importing the Gabor stage.
"""
