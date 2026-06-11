#!/usr/bin/env python3
"""Torch-free, librosa-free 128-mel features matching NeMo's preprocessor.

numpy-only (no librosa/scipy/numba) to keep the host-RAM footprint tiny. The
slaney mel filterbank and the (n_fft-padded, periodic-Hann) window are
precomputed once into assets/*.npy. Replicates NeMo AudioToMelSpectrogramPreprocessor:
preemph 0.97, STFT n_fft=512 win=400 hop=160 center/reflect, power spectrum,
128-mel 0..8000 Hz, natural-log with 2^-24 guard, normalize="NA".
"""
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
N_FFT = 512
HOP = 160
PREEMPH = 0.97
LOG_GUARD = 2.0 ** -24

_MEL_FB = np.load(os.path.join(_HERE, "assets", "mel_fb.npy"))   # (128, 257)
_WINDOW = np.load(os.path.join(_HERE, "assets", "window.npy"))   # (512,)


def _stft_power(x):
    """librosa-compatible STFT magnitude^2: center+reflect pad, framed, windowed rfft."""
    x = np.pad(x, N_FFT // 2, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP]  # (T, 512)
    spec = np.fft.rfft(frames * _WINDOW, axis=1)                        # (T, 257)
    return (np.abs(spec) ** 2).T.astype(np.float32)                    # (257, T)


def mel(samples: np.ndarray) -> np.ndarray:
    """samples: float32 mono 16k in [-1,1]. Returns (128, T) float32 log-mel."""
    if samples.size <= N_FFT // 2:
        # Too short for the reflect pad (np.pad mode="reflect" raises on empty
        # input — seen live when VAD fires a stop with no audio buffered).
        # <=16 ms holds no speech; an empty mel makes feed/finalize no-ops.
        return np.zeros((128, 0), np.float32)
    x = samples.astype(np.float32)
    x = np.concatenate([x[:1], x[1:] - PREEMPH * x[:-1]]).astype(np.float32)  # preemph
    power = _stft_power(x)                                              # (257, T)
    melspec = _MEL_FB @ power                                          # (128, T)
    return np.log(melspec + LOG_GUARD).astype(np.float32)
