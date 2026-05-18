# PyGaborSTM

Python library for extracting **Rate-Scale-Frequency (RSF)** representations
from audio signals using bio-inspired auditory spectrograms and 2D Gabor
filterbanks, following Chi, Ru & Shamma (2005) and Bellur & Elhilali (2017).

## What it does

```text
audio ──▶ AuditorySpectrogram ──▶ GaborFilterbank ──▶ RSF
              (n_freq × n_time)         (n_frames × n_rates × n_scales × n_freq)
```

- **CPU or GPU**: drop-in NumPy/CuPy backend.
- **Memory-adaptive Gabor stage**: caches kernel FFTs when memory allows,
  falls back to streaming otherwise.
- **Custom batched-SOS CUDA kernel** for the cochlear filter stage.

## Where to go next

- [Getting Started](getting-started.md) — install and run the pipeline on
  one audio file.
- [API Reference](api/index.md) — generated from the source docstrings.
- [GitHub repository](https://github.com/JHU-LCAP/PyGaborSTM) — issues, PRs.

## References

- Bellur, A., & Elhilali, M. (2017). Feedback-driven sensory mapping
  adaptation for robust speech activity detection. *IEEE/ACM Transactions
  on Audio, Speech, and Language Processing*, 25(3), 481–492.
- Chi, T., Ru, P., & Shamma, S. A. (2005). Multiresolution spectrotemporal
  analysis of complex sounds. *The Journal of the Acoustical Society of
  America*, 118(2), 887–906.
