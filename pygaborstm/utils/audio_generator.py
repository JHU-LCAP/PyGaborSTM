"""
Stimulus generation for auditory spectrogram validation.

Based on Chi, Ru & Shamma (2005) "Multiresolution spectrotemporal analysis of complex sounds"
"""
import numpy as np
from pathlib import Path
import soundfile as sf

# Defaults
SR = 16000
DURATION = 1.0

def generate_tone(freq, duration=DURATION, sr=SR, amplitude=0.5):
    """Generate a pure Sine tone."""
    t = np.arange(int(duration * sr)) / sr
    return amplitude * np.sin(2 * np.pi * freq * t)


def generate_three_tones(duration=DURATION, sr=SR):
    """Generate 250, 1000, 4000 Hz tones (Chi 2005 Section III.B.1)."""
    return (
        generate_tone(250, duration, sr),
        generate_tone(1000, duration, sr),
        generate_tone(4000, duration, sr),
    )


def generate_broadband_noise(duration=DURATION, sr=SR, seed=42):
    """Generate broadband noise - 59 random-phase tones (Chi 2005 Section III.B.2)."""
    rng = np.random.default_rng(seed)
    freqs = np.logspace(np.log2(135), np.log2(7465), 59, base=2.0)
    phases = rng.uniform(0, 2 * np.pi, 59)
    
    t = np.arange(int(duration * sr)) / sr
    signal = sum(np.sin(2 * np.pi * f * t + p) for f, p in zip(freqs, phases))
    return 0.5 * signal / np.max(np.abs(signal))


def generate_harmonic_complex(f0=80, duration=DURATION, sr=SR, phase_type="in_phase", seed=42):
    """Generate harmonic complex F0=80 Hz (Chi 2005 Section III.B.3)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    signal = np.zeros_like(t)
    
    for n in range(1, 51):
        freq = f0 * n
        if freq >= sr / 2:
            break
        phase = 0.0 if phase_type == "in_phase" else rng.uniform(0, 2 * np.pi)
        signal += np.sin(2 * np.pi * freq * t + phase)
    
    return 0.5 * signal / np.max(np.abs(signal))


def generate_moving_ripple(rate, scale, duration=DURATION, sr=SR,
                           mod_depth=0.9, f0=1000, bandwidth=5.3, df=1/16):
    """
    Generate moving ripple (spectrotemporally modulated noise).
    
    Matches MATLAB mvripfft function parameters.
    
    Args:
        rate: Temporal modulation rate ω (Hz), negative = downward
        scale: Spectral modulation scale Ω (cycles/octave)
        duration: Duration in seconds
        sr: Sample rate
        mod_depth: Modulation depth Am (0-1), default 0.9
        f0: Center frequency (Hz), default 1000
        bandwidth: Bandwidth in octaves, default 5.3
        df: Frequency spacing in octaves, default 1/16
    """
    # Frequency axis (log-spaced, matching MATLAB)
    n_freqs = int(bandwidth / df)
    x = np.linspace(-bandwidth/2, bandwidth/2, n_freqs)  # octaves from f0
    freqs = f0 * (2 ** x)
    
    # Time axis
    n_samples = int(duration * sr)
    t = np.arange(n_samples) / sr
    
    # Random phases for each frequency component
    rng = np.random.default_rng()
    phases = rng.uniform(0, 2 * np.pi, n_freqs)
    
    # Generate ripple: sum of modulated sinusoids
    signal = np.zeros(n_samples)
    for i, (freq, xi, phi) in enumerate(zip(freqs, x, phases)):
        # Ripple envelope: 1 + Am * sin(2π * Ω * x + 2π * ω * t)
        envelope = 1 + mod_depth * np.sin(2 * np.pi * scale * xi + 2 * np.pi * rate * t)
        carrier = np.sin(2 * np.pi * freq * t + phi)
        signal += envelope * carrier
    
    return signal / np.max(np.abs(signal))


def generate_ripple_set(output_dir, rates=None, scales=None, duration=DURATION, sr=SR):
    """
    Generate full set of ripple stimuli.
    
    Default: 10 rates × 6 scales = 60 ripples (matching MATLAB script)
    """
    if rates is None:
        rates = [-32, -16, -8, -4, -2, 2, 4, 8, 16, 32]
    if scales is None:
        scales = [0.25, 0.5, 1, 2, 4, 8]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {len(rates)} × {len(scales)} = {len(rates) * len(scales)} ripples...")
    
    counter = 1
    for rate in rates:
        for scale in scales:
            ripple = generate_moving_ripple(rate, scale, duration, sr)
            filename = f"ripple_{counter:02d}_R{rate:.2f}_S{scale:.2f}.wav"
            sf.write(output_dir / filename, ripple, sr)
            counter += 1
    
    print(f"Saved {counter - 1} ripples to {output_dir}")


# === Save functions ===

def save_three_tones(output_dir, duration=DURATION, sr=SR):
    """Save three test tones."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for freq, audio in zip([250, 1000, 4000], generate_three_tones(duration, sr)):
        sf.write(output_dir / f"tone_{freq}Hz.wav", audio, sr)


def save_noise(output_dir, duration=DURATION, sr=SR, seed=42):
    """Save broadband noise."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(output_dir / "broadband_noise.wav", generate_broadband_noise(duration, sr, seed), sr)


def save_harmonic_complexes(output_dir, f0=80, duration=DURATION, sr=SR, seed=42):
    """Save harmonic complexes (in-phase and random-phase)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for phase_type in ["in_phase", "random_phase"]:
        audio = generate_harmonic_complex(f0, duration, sr, phase_type.replace("_phase", ""), seed)
        sf.write(output_dir / f"harmonic_complex_F0{int(f0)}_{phase_type}.wav", audio, sr)
