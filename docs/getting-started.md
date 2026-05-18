# Getting Started

## Install

PyGaborSTM is not yet on PyPI. Install from source:

```bash
git clone https://github.com/JHU-LCAP/PyGaborSTM.git
cd PyGaborSTM
poetry install
```

### GPU support (optional)

GPU acceleration uses CuPy. On Linux/Windows with a CUDA-capable NVIDIA
GPU and the matching CUDA toolkit installed, `poetry install` will pull
the correct `cupy-cudaXXx` wheel. The library automatically falls back
to NumPy with a `UserWarning` if CuPy is unavailable.

```bash
# Check your CUDA version
nvidia-smi
```

| CUDA version | CuPy wheel       |
|--------------|------------------|
| 11.x         | `cupy-cuda11x`   |
| 12.x         | `cupy-cuda12x`   |
| 13.x         | `cupy-cuda13x`   |

## Quick start

```python
import pygaborstm as stm

# CPU
model = stm.PyGaborSTM()

# GPU (or fall back to CPU with a warning)
model = stm.PyGaborSTM(config=stm.Config(use_gpu=True))

# Two-stage usage
spec = model.spectrogram(audio)   # Spectrogram dataclass
rsf  = model.rsf(spec)            # RSF dataclass

# Or chain both stages on device with no intermediate host copy
rsf = model.compute(audio)

# Visualize
stm.plot.plt_spectrogram(spec)
stm.plot.plt_rsf(rsf)
stm.plot.plt_rsf(rsf, fold=True)  # symmetric folding
```

See [`notebooks/example_usage.ipynb`](https://github.com/JHU-LCAP/PyGaborSTM/blob/main/notebooks/example_usage.ipynb)
in the repo for a worked example.

## Configuration

All pipeline parameters live in a single [`Config`](api/config.md) dataclass:

```python
config = stm.Config(
    use_gpu=False,          # GPU acceleration
    sample_rate=16000,      # audio sample rate
    n_filters=128,          # cochlear channels
    f_min=180.0,            # lowest filter center frequency (Hz)
    octaves=5.3,            # frequency range in octaves
    resolution="low",       # "low", "medium", "high", "ultra", "max", "overkill"
)
```

See the [Config reference](api/config.md) for the full list of fields and
their defaults.
