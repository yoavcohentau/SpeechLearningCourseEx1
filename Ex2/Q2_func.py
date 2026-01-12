import numpy as np
from scipy.signal import stft, istft


def compute_steering_vector(mic_positions, source_pos, freqs, ref_mic_index=None, c=343):
    I = len(mic_positions)

    # mics positions relative to the reference mic
    q = np.array([np.linalg.norm(mic_pos - source_pos) for mic_pos in mic_positions])
    if ref_mic_index is not None:
        q = q - q[ref_mic_index]

    # Compute d(f)
    num_freqs = len(freqs)
    d = np.zeros((num_freqs, I), dtype=np.complex64)
    for i in range(I):
        d[:, i] = np.exp(-2j * np.pi * freqs * q[i] / c)

    return d


def apply_dsb(mic_signals, fs, mic_positions, source_pos, ref_mic_index):
    # params
    num_mics = mic_signals.shape[0]
    nperseg = 512
    noverlap = 256

    # STFT for each channel
    stft_list = []  # (num_mics, f, t)
    for m in range(num_mics):
        f, t, stft_vals = stft(mic_signals[m], fs=fs, nperseg=nperseg, noverlap=noverlap)
        stft_list.append(stft_vals)

    x = np.stack(stft_list, axis=-1)  # transpose to (f, t, mics)

    # steering vector
    d = compute_steering_vector(mic_positions, source_pos, f, ref_mic_index)

    # DSB weights
    w = d / num_mics
    w_conj = np.conj(w.reshape(len(f), 1, num_mics))

    # apply beamformer: z(t, f) = w^H(f) * x(t, f)
    z = np.sum(w_conj * x, axis=2)  # (freqs, time)

    # ISTFT
    _, out_signal = istft(z, fs=fs, nperseg=nperseg, noverlap=noverlap)

    return out_signal
