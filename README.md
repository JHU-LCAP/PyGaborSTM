# PyGaborSTM
PyGaborSTM is a Python library for extracting Rate-Scale-Frequency (RSF) representations from audio signals using bio-inspired auditory spectrograms and 2D Gabor filterbanks.

## Installation

<!-- TODO: Publish to PyPI -->
```bash
pip install pygaborstm
```

### From source
```bash
git clone https://github.com/JHU-LCAP/PyGaborSTM.git
cd pygaborstm
poetry install
```

## Quick Start
```python
import pygaborstm as stm

# One-liner
rsf = stm.compute_rsf("audio.wav")

# Step by step
audio, sr = stm.load("audio.wav")
spectrogram = stm.auditory_spectrogram(audio)
rsf = stm.rsf(spectrogram)

# Visualization
stm.plot_spectrogram(spectrogram)
stm.plot_rsf(rsf)              # Unfolded
stm.plot_rsf(rsf, fold=True)   # Symmetric

# Access data
rs_matrix = rsf.rate_scale_matrix()  # For visualization
rsf_3d = rsf.mean_over_time()        # For TSVD input
```

See `notebooks/example_usage.ipynb` for more examples.

## Directory Structure
```
pygaborstm/
├── pygaborstm/
│   ├── __init__.py      # Public API
│   ├── config.py        # SpectrogramConfig, GaborConfig, Config
│   ├── structs.py       # Spectrogram, RSF dataclasses
│   ├── spectrogram.py   # AuditorySpectrogram
│   ├── gabor.py         # GaborFilterbank
│   ├── core.py          # load(), compute_rsf()
│   └── plotting.py      # plot_spectrogram(), plot_rsf(), plot_filterbank()
├── notebooks/
│   ├── assets/
│   └── example_usage.ipynb
└── tests/
```

## Development
```bash
poetry run jupyter notebook  # Run notebooks
poetry run pytest -v         # Run tests
```

## References
- Bellur, A., & Elhilali, M. (2017). Feedback-driven sensory mapping adaptation for robust speech activity detection. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 25(3), 481-492.