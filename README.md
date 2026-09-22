# PyGaborSTM

[![PyPI](https://img.shields.io/pypi/v/pygaborstm?cacheSeconds=3600)](https://pypi.org/project/pygaborstm/)
[![CI](https://github.com/JHU-LCAP/PyGaborSTM/actions/workflows/ci.yml/badge.svg)](https://github.com/JHU-LCAP/PyGaborSTM/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/pygaborstm/badge/?version=latest)](https://pygaborstm.readthedocs.io/en/latest/)
[![License](https://img.shields.io/pypi/l/pygaborstm?cacheSeconds=3600)](LICENSE)

PyGaborSTM is a Python library for extracting Rate-Scale-Frequency (RSF) representations from audio signals using bio-inspired auditory spectrograms and 2D Gabor filterbanks. Documentation can be found [here](https://pygaborstm.readthedocs.io/en/latest/).

The project is `PyGaborSTM`; the package you install and import is lowercase
`pygaborstm`.

> **Status:** 0.1.0, a first public release. The API may still change in
> backwards-incompatible ways before 1.0.

## Installation

```bash
pip install pygaborstm          # pip
poetry add pygaborstm           # poetry
```

This is the CPU install and works on macOS, Linux and Windows. It pulls only
NumPy, SciPy and psutil.

Plotting (`pygaborstm.plot` and `pygaborstm.analysis`) needs matplotlib, which
is an extra so the core install stays small:

```bash
pip install 'pygaborstm[viz]'
poetry add "pygaborstm[viz]"
```

### GPU (optional, NVIDIA only)
Pick the extra matching your CUDA version, which `nvidia-smi` reports:

```bash
pip install 'pygaborstm[cuda12]'   # CUDA 12.x
pip install 'pygaborstm[cuda13]'   # CUDA 13.x

poetry add "pygaborstm[cuda12]"
poetry add "pygaborstm[cuda13]"
```

Install only one: both provide the `cupy` module. The extra also pulls the
CUDA libraries from PyPI (about 1.6 GB), so only the NVIDIA driver needs to be
installed; if you already have a CUDA toolkit and want to skip that download,
install `cupy-cuda13x` (or `cupy-cuda12x`) directly instead of the extra. CuPy
has no macOS wheels, so on macOS these extras install nothing and the library
runs on CPU.

Then set `use_gpu=True`:

```python
model = stm.PyGaborSTM(stm.Config(use_gpu=True))
```

If CuPy is missing or no CUDA device is present, the library warns and falls
back to NumPy rather than failing. Check what it actually resolved with
`model.device.on_gpu`.

### From source
```bash
git clone https://github.com/JHU-LCAP/PyGaborSTM.git
cd PyGaborSTM
poetry install
```

### Depending on unreleased work

Nothing is published between releases. To build against `main`, pin a commit:

```toml
# your pyproject.toml
pygaborstm = { git = "https://github.com/JHU-LCAP/PyGaborSTM.git", rev = "<sha>" }
```

Between releases the version reads `X.Y.Z.dev0` for every commit, so it tells
you which release is being worked toward, not which commit you have. The commit
is recorded by pip at install time and read back with:

```python
>>> import pygaborstm as stm
>>> stm.build_info()
{'version': '0.1.1.dev0', 'source': 'git', 'commit': 'a1b2c3d...', 'url': '...'}
```

This works even if you pinned a branch: pip records the commit it resolved.
An install from PyPI reports `source: 'index'` and no commit, because there the
version identifies the build.

To see what a pinned commit contains relative to the last release:

```bash
git log --oneline v0.1.0..<sha>
```

## Quick Start
```python
import numpy as np
import pygaborstm as stm

# Any mono 1-D float array at Config.sample_rate (16 kHz by default).
# It is mean-removed and peak-normalised before analysis.
sr = 16000
t = np.arange(2 * sr) / sr
audio = np.sin(2 * np.pi * 440 * t)

model = stm.PyGaborSTM()                              # CPU
# model = stm.PyGaborSTM(stm.Config(use_gpu=True))    # NVIDIA GPU

spec = model.spectrogram(audio)   # (n_freq, n_time)
rsf = model.rsf(spec)             # (n_frames, n_rates, n_scales, n_freq)
rsf = model.compute(audio)        # both stages, no intermediate host copy

# Plotting needs the [viz] extra.
stm.plot.plt_spectrogram(spec)
stm.plot.plt_rsf(rsf)
stm.plot.plt_rsf(rsf, fold=True)  # symmetric folding
```

See `notebooks/example_usage.ipynb` for more examples.

### Notebooks and their audio

The notebooks are not self-contained: `data/` and `notebooks/assets/` are
gitignored, so the audio they read is absent from a fresh clone.

- `chi2005_validation.ipynb` regenerates most of its own stimuli, so it runs
  after executing its generation cell.
- `example_usage.ipynb` and `benchmark.ipynb` need audio you supply; point them
  at any mono file.
- `chi99_validation.ipynb` and `mvripfft_validation.ipynb` need the ripple set,
  which `notebooks/nb_utils/audio_generator.py:generate_ripple_set` produces.
  Note the default rates there are narrower than the set `chi99_validation`
  expects, so it reproduces the method rather than the exact figure.

## Configuration
```python
config = stm.Config(
    # General
    use_gpu=False,          # Enable GPU acceleration
    sample_rate=16000,      # Audio sample rate
    
    # Spectrogram
    n_filters=128,          # Number of frequency channels
    f_min=180.0,            # Minimum frequency (Hz)
    octaves=5.3,            # Frequency range in octaves
    
    # RSF / Gabor
    resolution="low",       # "low", "medium", "high", "ultra", "max", "overkill"
)
```

## Directory Structure
```
PyGaborSTM/
├── pygaborstm/
│   ├── __init__.py      # Public API
│   ├── config.py        # Config dataclass
│   ├── constants.py     # Standard rate/scale grids, resolution multipliers
│   ├── structs.py       # Spectrogram, RSF dataclasses
│   ├── spectrogram.py   # AuditorySpectrogram
│   ├── gabor.py         # GaborFilterbank
│   ├── core.py          # PyGaborSTM class
│   ├── plot.py          # Plotting functions (needs [viz])
│   ├── analysis.py      # MTF analysis helpers (needs [viz])
│   ├── backend.py       # Device resolution, NumPy/CuPy switching
│   ├── _optional.py     # Optional-dependency errors
│   └── gammatone_kernel.py  # Custom CUDA SOS kernel
├── notebooks/
└── tests/
```

## Development
```bash
poetry install                      # Install all dependencies
poetry run jupyter notebook         # Run notebooks
poetry run pytest -m "not gpu"      # Run all tests excluding GPU kernel tests (used in CI/CD)
poetry run pytest -v                # Run all tests including GPU kernel tests
poetry run ruff check --fix .       # lint and fix
poetry run ruff format .            # format code
```

### Serve Docs locally
```bash
poetry run mkdocs serve
```

Note: Please lint and format before pushing, as CI will fail otherwise.

### Jupyter Kernel
Ensure your notebook uses the correct Poetry environment:
```bash
# Check Poetry env path
poetry env info --path

# Register kernel (if needed)
poetry run python -m ipykernel install --user --name pygaborstm
```

## Citing

If you use PyGaborSTM in published work, please cite the software (see
`CITATION.cff`, or the "Cite this repository" button on GitHub) along with the
papers it implements.

## References
- Chi, T., Ru, P., & Shamma, S. A. (2005). Multiresolution spectrotemporal analysis of complex sounds. *The Journal of the Acoustical Society of America*, 118(2), 887-906. [doi:10.1121/1.1945807](https://doi.org/10.1121/1.1945807)

  The auditory spectrogram model implemented here.

- Bellur, A., & Elhilali, M. (2017). Feedback-driven sensory mapping adaptation for robust speech activity detection. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 25(3), 481-492. [doi:10.1109/TASLP.2016.2639322](https://doi.org/10.1109/TASLP.2016.2639322)

  The Gabor filterbank formulation used for the RSF stage.
