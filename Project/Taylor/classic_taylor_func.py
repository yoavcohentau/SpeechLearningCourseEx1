import numpy as np
import librosa

from Ex2.Q2_func import apply_mvdr


def taylor_classic_with_known_noise(mic_signals, noise_reference, fs=16000):
    """
    Improved TaylorBeamformer using known noise as the explicit delta.
    Incorporates Power Compression for better residual cancellation.
    """
    # 1. 0th order: Standard MVDR
    s_0_spec, stft_params = apply_mvdr(mic_signals, noise_reference, is_return_stft=True, use_taylor=True)

    # 2. Extract Noise STFT (This is our EXACT delta)
    # The paper defines delta as the interference to be cancelled.
    # Using the noise_reference directly makes the approximation highly accurate.
    noise_stft = librosa.stft(noise_reference[0], n_fft=stft_params[2],
                              hop_length=stft_params[0], win_length=stft_params[1])

    # 3. Power Compression (Factor of 0.5 as suggested in paper)
    # This stabilizes the high-order terms and helps in dereverberation.
    def compress(x): return np.abs(x) ** 0.5 * np.exp(1j * np.angle(x))

    def decompress(x): return np.abs(x) ** 2.0 * np.exp(1j * np.angle(x))

    s0_comp = compress(s_0_spec)
    delta_comp = compress(noise_stft)

    # 4. High-order derivative operators
    # We estimate the derivative of the restoration process relative to the input.
    # Since we use known noise, T1 and T2 now act as active cancellers.
    first_order_deriv = np.diff(s0_comp, axis=1, append=s0_comp[:, -1:])
    second_order_deriv = np.diff(first_order_deriv, axis=1, append=first_order_deriv[:, -1:])

    # 5. Taylor Superimposition (Eqn. 19)
    # S = S0 + T1 + 0.5*T2
    # We subtract the terms because delta (noise) is what we want to remove.
    alpha = 0.4  # Increased weight since delta is now accurate
    beta = 0.2

    t1 = alpha * first_order_deriv * delta_comp
    t2 = beta * second_order_deriv * (delta_comp ** 2)

    # Superimpose high-order corrections onto the spatial base
    enhanced_comp = s0_comp - (t1 + t2)

    # 6. Decompress and reconstruct
    enhanced_spec = decompress(enhanced_comp)

    return librosa.istft(
        enhanced_spec,
        hop_length=stft_params[0],
        win_length=stft_params[1],
        n_fft=stft_params[2],
        length=stft_params[3]
    )


def taylor_classic_with_orders_and_delta(mic_signals, white_noise_scaled, fs=16000):
    """
    Classic TaylorBeamformer implementation using explicit delta (residual error).
    S = S0 + (dG/dX)*delta + 0.5*(d2G/dX2)*delta^2
    """
    # 1. 0th order: Standard MVDR (The spatial base)
    s_0_spec, stft_params = apply_mvdr(mic_signals, white_noise_scaled, is_return_stft=True)

    # 2. Reference channel STFT (The 'X' in our Taylor expansion)
    x_ref_stft = librosa.stft(mic_signals[0], n_fft=stft_params[2],
                              hop_length=stft_params[0], win_length=stft_params[1])

    # 3. Explicit Delta calculation
    # delta = X - S0 (The interference/residual noise we want to cancel)
    delta = x_ref_stft - s_0_spec

    # 4. Derivative Operators (Approximated by spectral/temporal differences)
    # These represent the change in the restoration function G
    first_order_deriv = np.diff(s_0_spec, axis=1, append=s_0_spec[:, -1:])
    second_order_deriv = np.diff(first_order_deriv, axis=1, append=first_order_deriv[:, -1:])

    # 5. Taylor Superimposition (Eqn. 19)
    # S_final = S0 + T1 + T2
    # T1 = (dG/dX) * delta
    # T2 = 0.5 * (d2G/dX2) * delta^2

    # We use small scaling weights (alpha, beta) to stabilize the classical approximation
    alpha = 0.2
    beta = 0.1

    t1 = alpha * first_order_deriv * delta
    t2 = beta * second_order_deriv * (delta ** 2)

    # Combining all terms to recover the clean signal
    enhanced_spec = s_0_spec + t1 + t2

    # 6. Final reconstruction to time domain
    return librosa.istft(
        enhanced_spec,
        hop_length=stft_params[0],
        win_length=stft_params[1],
        n_fft=stft_params[2],
        length=stft_params[3]
    )


def taylor_classic_with_orders(mic_signals, white_noise_scaled, fs=16000):
    # 0th order: Standard MVDR [cite: 111, 256]
    s_0_spec, stft_params = apply_mvdr(mic_signals, white_noise_scaled, is_return_stft=True)

    # 1st and 2nd orders approximation [cite: 12, 114, 150]
    # We use temporal differences to model the derivative operators
    first_order_diff = np.diff(s_0_spec, axis=1, append=s_0_spec[:, -1:])
    second_order_diff = np.diff(first_order_diff, axis=1, append=first_order_diff[:, -1:])

    # Taylor Superimposition (Eqn. 19): S = S0 + T1 + 0.5*T2 [cite: 150]
    # In classical terms, this acts as a dynamic smoother/de-reverberator
    enhanced_spec = s_0_spec + 0.5 * first_order_diff + 0.25 * second_order_diff

    return librosa.istft(
        enhanced_spec,
        hop_length=stft_params[0],
        win_length=stft_params[1],
        n_fft=stft_params[2],
        length=stft_params[3]
    )


# def taylor_classic_wrapper(mic_signals, white_noise_scaled, fs=16000):
#     """
#     Classic implementation of TaylorBeamformer.
#     Separates spatial filtering (0th order) from spectral refinement (high order).
#     """
#     # 1. Obtain the 0th-order term (Your MVDR)
#     # The paper formulates this as the initial spatial filtering[cite: 12, 111].
#     s_0_spec, stft_params = apply_mvdr(mic_signals, white_noise_scaled, is_return_stft=True)
#
#     # Extract reference channel STFT for high-order guidance [cite: 141]
#     x_ref_stft = librosa.stft(mic_signals[0], n_fft=stft_params[2],
#                               hop_length=stft_params[0], win_length=stft_params[1])
#
#     # 2. Power Compression (Critical for stability and audibility)
#     # The paper uses a compression factor of 0.5[cite: 188].
#     def compress(x):
#         return np.abs(x) ** 0.5 * np.exp(1j * np.angle(x))
#
#     def decompress(x):
#         return np.abs(x) ** 2.0 * np.exp(1j * np.angle(x))
#
#     s0_comp = compress(s_0_spec)
#     x_ref_comp = compress(x_ref_stft)
#
#     freq_bins, frames = s0_comp.shape
#     enhanced_comp = np.zeros_like(s0_comp)
#
#     # 3. High-order Module (Residual Noise Canceller)
#     # This acts as the spectral canceller to suppress residual/diffuse noise[cite: 114, 257].
#     alpha_noise = 0.9  # Smoothing factor
#
#     for f in range(freq_bins):
#         # We estimate the 'delta' (interference) as the difference between
#         # the mixture and the spatially filtered output[cite: 102, 108].
#         delta_f = x_ref_comp[f, :] - s0_comp[f, :]
#
#         # Track residual noise power in the frequency bin
#         res_noise_pow = np.mean(np.abs(delta_f) ** 2)
#
#         for t in range(frames):
#             # Calculate local SNR after spatial filtering
#             p_s0 = np.abs(s0_comp[f, t]) ** 2
#             p_delta = np.abs(delta_f[t]) ** 2
#
#             # Taylor refinement logic: S = S0 + T(1)
#             # T(1) is a correction term aimed at cancelling the remaining interference.
#             # Here we model T(1) as a subtraction of the residual 'delta' scaled by noise confidence.
#             snr_local = p_s0 / (p_delta + 1e-10)
#
#             # The correction factor (proxy for the learnable derivative)
#             # In high SNR, correction is near 0. In low SNR, it cancels more[cite: 37, 114].
#             t1_weight = 1.0 / (snr_local + 1.5)
#             t1_correction = -0.7 * t1_weight * delta_f[t]
#
#             # Superimposition (Eqn. 19): Add the high-order correction to the 0th-order base
#             enhanced_comp[f, t] = s0_comp[f, t] + t1_correction
#
#     # 4. Final Reconstruction
#     # Decompress and apply ISTFT
#     enhanced_spec = decompress(enhanced_comp)
#
#     out_taylor = librosa.istft(
#         enhanced_spec,
#         hop_length=stft_params[0],
#         win_length=stft_params[1],
#         n_fft=stft_params[2],
#         length=stft_params[3]
#     )
#
#     return np.real(out_taylor)

def taylor_classic_wrapper(noisy_white, white_noise_scaled, fs=16000):
    """
    Updated Taylor wrapper using true superimposition and power compression.
    """
    # 1. Power Compression (Source 188 suggests factor of 0.5)
    # This helps stabilize the estimation of high-order terms.
    compress = lambda x: np.abs(x) ** 0.5 * np.exp(1j * np.angle(x))
    decompress = lambda x: np.abs(x) ** 2.0 * np.exp(1j * np.angle(x))

    # 0th-order term (Spatial Filtering) [cite: 111, 113]
    s_0_spec, stft_params = apply_mvdr(noisy_white, white_noise_scaled, is_return_stft=True)

    # Input X for high-order guidance [cite: 141]
    x_ref_stft = librosa.stft(noisy_white[0], n_fft=stft_params[2],
                              hop_length=stft_params[0], win_length=stft_params[1])

    # Apply compression to the spectra for processing
    s_0_comp = compress(s_0_spec)
    x_ref_comp = compress(x_ref_stft)

    freq_bins, frames = s_0_comp.shape
    enhanced_comp = np.zeros_like(s_0_comp)

    for f_idx in range(freq_bins):
        s_0_bin = s_0_comp[f_idx, :]
        x_ref_bin = x_ref_comp[f_idx, :]

        # 2. Estimate T(1) - The first-order derivative term [cite: 119]
        # Instead of gain, we calculate a 'Correction Signal' that targets residual noise.
        # In the paper, the high-order terms are supervised as residual cancellers[cite: 163].

        # Calculate the difference (residual noise proxy)
        # delta = X - S (approximating the interference to be cancelled)
        delta_bin = x_ref_bin - s_0_bin

        # Smoothing factor for high-order tracking
        alpha = 0.8
        noise_floor = np.zeros(frames)
        curr_noise = np.abs(delta_bin[0]) ** 2

        for t in range(frames):
            curr_noise = alpha * curr_noise + (1 - alpha) * np.abs(delta_bin[t]) ** 2
            noise_floor[t] = curr_noise

        # 3. Taylor Superimposition (Eqn. 19): S = S0 + sum( (1/q!) * T(q) )
        # T(1) is modeled as a correction that subtracts the estimated residual
        # interference, scaled by the confidence in the 0th-order result.

        snr_est = np.abs(s_0_bin) ** 2 / (noise_floor + 1e-10)

        # The 'Correction' term (T1 approximation)
        # As SNR decreases, the high-order term becomes more active to cancel noise[cite: 114].
        t1_correction = -1.0 * delta_bin * (1 / (snr_est + 1.2))

        # Perform actual superimposition as per the paper
        enhanced_comp[f_idx, :] = s_0_bin + t1_correction

    # Decompress back to linear power
    enhanced_spec = decompress(enhanced_comp)

    # Final reconstruction
    out_taylor = librosa.istft(
        enhanced_spec,
        hop_length=stft_params[0],
        win_length=stft_params[1],
        n_fft=stft_params[2],
        length=stft_params[3]
    )

    return np.real(out_taylor)


# def taylor_classic_wrapper(noisy_white, white_noise_scaled, fs=16000):
#     # 1. Obtain the 0th-order term (Spatial Filtering)
#     # out_stft shape: (freq_bins, frames)
#     s_0_spec, stft_params = apply_mvdr(noisy_white, white_noise_scaled, is_return_stft=True)
#
#     # Get the STFT of the reference microphone (usually the first one)
#     # This is crucial because Taylor high-order terms depend on the input X
#     x_ref_stft = librosa.stft(noisy_white[0], n_fft=stft_params[2],
#                               hop_length=stft_params[0], win_length=stft_params[1])
#
#     freq_bins, frames = s_0_spec.shape
#     enhanced_spec = np.zeros_like(s_0_spec)
#
#     # 2. Estimate the Residual Interference (The "delta" component)
#     # In Taylor theory, delta = -interference .
#     # We compare the input X to the spatial filtered output S0.
#     for f_idx in range(freq_bins):
#         s_0_bin = s_0_spec[f_idx, :]
#         x_ref_bin = x_ref_stft[f_idx, :]
#
#         # The 'Interference' that MVDR tried to remove
#         # Classically, this difference is a proxy for the residual noise
#         interference_approx = x_ref_bin - s_0_bin
#
#         psd_s0 = np.abs(s_0_bin) ** 2
#         psd_interference = np.abs(interference_approx) ** 2
#
#         # 3. High-order Correction (Taylor Superimposition)
#         # Instead of a simple Wiener gain, we use the Taylor expansion
#         # logic: S = S0 + T(1) + T(2)...
#         # T(1) is a derivative-based correction that cancels remaining noise
#
#         # Let's implement a more aggressive 'Residual Canceller' gain
#         # This gain is applied to S0 to further suppress the interference PSD found in X
#         snr_residual = psd_s0 / (psd_interference + 1e-10)
#
#         # Taylor-like refinement:
#         # If SNR is low, the high-order terms should act as a spectral subtractor
#         gain = np.clip(1 - (1.0 / (snr_residual + 0.5)), 0.01, 1.0)
#
#         # Superimpose the correction
#         enhanced_spec[f_idx, :] = s_0_bin * gain
#
#     # Final reconstruction
#     out_taylor = librosa.istft(
#         enhanced_spec,
#         hop_length=stft_params[0],
#         win_length=stft_params[1],
#         n_fft=stft_params[2],
#         length=stft_params[3]
#     )
#
#     return np.real(out_taylor)


# def taylor_classic_wrapper(noisy_white, white_noise_scaled, fs=16000):
#     """
#     Classic wrapper inspired by TaylorBeamformer.
#
#     mic_signals: Input array of shape (M, samples)
#     mvdr_function: External function that returns the beamformed spectrum (freq_bins, frames)
#     fs: Sampling frequency
#     """
#     # 1. Obtain the 0th-order term using your external MVDR function
#     # This represents the spatial filtering stage (Eqn. 9 in paper)
#     s_0_spec, stft_pararms = apply_mvdr(noisy_white, white_noise_scaled, is_return_stft=True)
#
#     freq_bins, frames = s_0_spec.shape
#     enhanced_spec = np.zeros_like(s_0_spec)
#
#     # 2. Classic estimation of residual noise (The "delta" component)
#     # In the paper, delta = -interference.
#     # Classically, we estimate the Power Spectral Density (PSD) of the interference.
#     for f_idx in range(freq_bins):
#         # Current spectrum of the beamformer output at this frequency
#         current_bin = s_0_spec[f_idx, :]
#
#         # Estimate Noise PSD using a smoothing window (Recursive averaging)
#         # This acts as a proxy for the high-order derivative terms' goal
#         alpha_s = 0.95
#         noise_psd = np.zeros(frames)
#         current_psd = np.abs(current_bin) ** 2
#
#         # Simple noise floor tracker
#         est_noise = current_psd[0]
#         for t in range(frames):
#             est_noise = alpha_s * est_noise + (1 - alpha_s) * current_psd[t]
#             noise_psd[t] = est_noise
#
#         # 3. Apply High-order Correction (Taylor Superimposition proxy)
#         # The paper uses derivatives to refine the spectrum.
#         # A classic equivalent is the Wiener Gain based on the estimated interference.
#         snr_post = current_psd / (noise_psd + 1e-10)
#
#         # Taylor expansion of (1 + x)^-1 is approximately 1 - x + x^2...
#         # Here we use a standard gain rule which matches the "residual canceller" logic
#         gain = np.maximum(0.1, 1 - (1 / (snr_post + 1)))
#
#         # Superimpose the correction onto the 0th-order result
#         enhanced_spec[f_idx, :] = current_bin * gain
#
#         out_taylor = librosa.istft(
#             enhanced_spec, hop_length=stft_pararms[0], win_length=stft_pararms[1], n_fft=stft_pararms[2], length=stft_pararms[3])
#
#     return out_taylor

# Example Usage:
# result_spec = taylor_classic_wrapper(my_5_mic_array, my_mvdr_func, 16000)