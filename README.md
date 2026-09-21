# PyGaborSTM
PyGaborSTM is a Python library for extracting Rate-Scale-Frequency (RSF) representations from audio signals using bio-inspired auditory spectrograms and 2D Gabor filterbanks. Documentation can be found [here](https://pygaborstm.readthedocs.io/en/latest/).

## Installation

```bash
pip install pygaborstm
```

This is the CPU install and works on macOS, Linux and Windows. It pulls only
NumPy, SciPy and psutil.

Plotting (`pygaborstm.plot` and `pygaborstm.analysis`) needs matplotlib, which
is an extra so the core install stays small:

```bash
pip install 'pygaborstm[viz]'
```

### GPU (optional, NVIDIA only)
Pick the extra matching your CUDA version, which `nvidia-smi` reports:

```bash
pip install 'pygaborstm[cuda12]'   # CUDA 12.x
pip install 'pygaborstm[cuda13]'   # CUDA 13.x
```

Install only one: both provide the `cupy` module. CuPy has no macOS wheels, so
on macOS these extras install nothing and the library runs on CPU.

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

### CUDA Toolkit
The CuPy wheels bundle the CUDA runtime, so a separate toolkit install is only
needed if you want `nvcc` and the profiling tools.

Get it from https://developer.nvidia.com/cuda-toolkit, then add to your shell
profile:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Quick Start
```python
import pygaborstm as stm

# Create model (CPU)
model = stm.PyGaborSTM()

# Create model (GPU)
model = stm.PyGaborSTM(config=stm.Config(use_gpu=True))

# Compute spectrogram and RSF
spec = model.spectrogram(audio)
rsf = model.rsf(spec)

# Visualization
stm.plot.plt_spectrogram(spec)
stm.plot.plt_rsf(rsf)
stm.plot.plt_rsf(rsf, fold=True)  # Symmetric folding
```

See `notebooks/example_usage.ipynb` for more examples.

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

## References
- Bellur, A., & Elhilali, M. (2017). Feedback-driven sensory mapping adaptation for robust speech activity detection. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 25(3), 481-492.
