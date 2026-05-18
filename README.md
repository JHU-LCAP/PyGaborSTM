# PyGaborSTM
PyGaborSTM is a Python library for extracting Rate-Scale-Frequency (RSF) representations from audio signals using bio-inspired auditory spectrograms and 2D Gabor filterbanks.

## Installation

<!-- TODO: Publish to PyPI -->
```bash
pip install pygaborstm # not published yet
```
For now, install from source (see below).

### From source
```bash
git clone https://github.com/JHU-LCAP/PyGaborSTM.git
cd PyGaborSTM
poetry install
```

### GPU Support (Optional, Linux/Windows only)
For GPU acceleration, you need:

1. **NVIDIA GPU** with CUDA support
2. **CUDA Toolkit** installed on your system

```bash
# Check your CUDA version
nvidia-smi
```

Download and install the CUDA Toolkit from NVIDIA:
https://developer.nvidia.com/cuda-toolkit

After installation, add to your `~/.bashrc` or `~/.zshrc`:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Verify installation:

```bash
nvcc --version
```

The library uses CuPy for GPU acceleration. Make sure your CuPy version matches your CUDA version:
- CUDA 11.x → `cupy-cuda11x`
- CUDA 12.x → `cupy-cuda12x`
- CUDA 13.x → `cupy-cuda13x`

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
stm.plot.spectrogram(spec)
stm.plot.rsf(rsf)
stm.plot.rsf(rsf, fold=True)  # Symmetric folding
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
    resolution="low",       # "low", "medium", "high", "ultra"
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
│   ├── plot.py          # Plotting functions
│   └── backend.py       # NumPy/CuPy switching
├── notebooks/
└── tests/
```

## Development
```bash
poetry install                      # Install all dependencies
poetry run jupyter notebook         # Run notebooks
pytest -m "not gpu"                 # Run all tests exluding GPU kernel tests (used in CI/CD)
poetry run pytest -v                # Run all tests including GPU kernel tests
poetry run ruff check --fix .       # lint and fix
poetry run ruff format .            # format code
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