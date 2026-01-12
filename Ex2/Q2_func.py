import numpy as np
from scipy.signal import stft, istft


def compute_steering_vector(mic_positions, source_pos, freqs, c=343):
    I = len(mic_positions)  # Number of microphones [cite: 48]

    # mics positions
    q = np.array([np.linalg.norm(mic_pos - source_pos) for mic_pos in mic_positions])

    # Initialize d matrix
    num_freqs = len(freqs)
    d = np.zeros((num_freqs, I), dtype=np.complex64)

    # Compute d(f) based on
    # The exponent is -2j * pi * q_i * f / c
    # This corresponds to the time delay q_i / c
    for i in range(I):
        d[:, i] = np.exp(-2j * np.pi * freqs * q[i] / c)

    # num_mics = len(mic_positions)
    # num_freqs = len(freqs)
    #
    # # Calculate distances from source to each mic and the delay
    # dists = np.array([np.linalg.norm(mic_pos - source_pos) for mic_pos in mic_positions])
    # delays = dists / c
    #
    # # steering vector relative to ref: d_i = exp(-j * w * tau_i).
    # d = np.zeros((num_freqs, num_mics), dtype=np.complex64)
    # omega = 2 * np.pi * freqs
    # for i in range(num_mics):
    #     d[:, i] = np.exp(-1j * omega * delays[i])

    return d


def apply_dsb(mic_signals, fs, mic_positions, source_pos, ref_mic_index):

    # params
    num_mics = mic_signals.shape[0]
    nperseg = 512
    noverlap = 256

    # 1. STFT
    # STFT for each channel
    stft_list = []  # (num_mics, f, t)
    for m in range(num_mics):
        f, t, stft_vals = stft(mic_signals[m], fs=fs, nperseg=nperseg, noverlap=noverlap)
        stft_list.append(stft_vals)

    # transpose to (f, t, mics)
    X = np.stack(stft_list, axis=-1)

    # 2. Compute Steering Vector
    d = compute_steering_vector(mic_positions, source_pos, f)

    # 3. Compute DSB Weights
    d_ref = d[:, ref_mic_index]  # Shape (freqs,)
    w = d / num_mics  # Shape (freqs, mics)

    # 4. Apply Beamformer: y(t, f) = w^H(f) * x(t, f)
    # X: (freqs, time, mics)
    # w: (freqs, mics) -> reshape to (freqs, 1, mics)
    # We want sum over mics of (conj(w) * X)

    w_conj = np.conj(w.reshape(len(f), 1, num_mics))
    Y_dry_source = np.sum(w_conj * X, axis=2)  # Result: (freqs, time)

    # Now, align to reference microphone (add back the delay of the reference)
    # This ensures the "speech" part aligns with what the reference mic heard.
    d_ref_reshaped = d[:, ref_mic_index].reshape(len(f), 1)
    Y_aligned = Y_dry_source * d_ref_reshaped

    # 5. ISTFT
    _, enhanced_signal = istft(Y_aligned, fs=fs, nperseg=nperseg, noverlap=noverlap)

    return enhanced_signal
