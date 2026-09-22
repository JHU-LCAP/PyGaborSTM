# Getting Started

## Install

```bash
pip install pygaborstm          # pip
poetry add pygaborstm           # poetry
```

This is the CPU install and works on macOS, Linux and Windows.
It pulls only NumPy, SciPy and psutil.

Plotting (`pygaborstm.plot` and `pygaborstm.analysis`) needs matplotlib, which is an extra so the core install stays small:

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

Install only one: both provide the `cupy` module.
The extra also pulls the CUDA libraries from PyPI (about 1.6 GB), so only the NVIDIA driver needs to be installed.
CuPy has no macOS wheels, so on macOS these extras install nothing and the library runs on CPU.

Then set `use_gpu=True`:

```python
model = stm.PyGaborSTM(stm.Config(use_gpu=True))
```

If CuPy is missing or no CUDA device is present, the library warns and falls back to NumPy rather than failing.
Check what it actually resolved with `model.device.on_gpu`.

### From source

```bash
git clone https://github.com/JHU-LCAP/PyGaborSTM.git
cd PyGaborSTM
poetry install
```

To depend on an unreleased commit from another project, and to read back which commit you got, see [Depending on unreleased work](https://github.com/JHU-LCAP/PyGaborSTM#depending-on-unreleased-work) in the README.

## Quick start

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

`spec` is a [`Spectrogram`](api/structs.md) and `rsf` an [`RSF`](api/structs.md); both carry their axes (`freqs`, `times`, `rates`, `scales`) alongside the data.

See [`notebooks/example_usage.ipynb`](https://github.com/JHU-LCAP/PyGaborSTM/blob/main/notebooks/example_usage.ipynb) in the repo for a worked example.

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

The RSF hop (`rsf_frame_shift_ms`) advances in whole spectrogram frames, so the default 10 ms request gives a 16 ms hop; `rsf.times` reports the actual one.

See the [Config reference](api/config.md) for the full list of fields and their defaults.
