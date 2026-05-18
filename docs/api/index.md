# API Reference

Auto-generated from the source docstrings.

## Top-level

- [PyGaborSTM](core.md) — main user-facing class.
- [Config](config.md) — configuration dataclass.
- [Spectrogram, RSF](structs.md) — output data structures.

## Pipeline stages

- [AuditorySpectrogram](spectrogram.md) — cochlear-model spectrogram (stage 1).
- [GaborFilterbank](gabor.md) — 2D Gabor filterbank + RSF extraction (stage 2).

## Visualization & analysis

- [Plotting](plot.md) — `matplotlib` helpers for spectrograms and RSFs.
- [Analysis](analysis.md) — matched-filter MTF computation.

## Internals

- [Backend helpers](backend.md) — NumPy/CuPy switching, memory probe, dtype pairs.
- [CUDA kernel](gammatone_kernel.md) — custom batched-SOS kernel for the y1 stage.
