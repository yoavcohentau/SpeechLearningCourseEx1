import numpy as np
import matplotlib.pyplot as plt
from rir_generator import generate
import librosa
import librosa.display
from scipy.signal import fftconvolve


def generate_room_impulse_responses(
        fs,
        room_dim,
        mic_center,
        num_mics,
        mic_spacing,
        source_angle_deg,
        source_distance,
        T60_values
):
    # microphone positions (on x-axis)
    mic_center = np.array(mic_center)
    mic_positions = [
        mic_center + np.array([(i - (num_mics - 1) / 2) * mic_spacing, 0, 0])
        for i in range(num_mics)
    ]

    # source position
    theta = np.deg2rad(source_angle_deg)
    source_pos = mic_center + source_distance * np.array([
        np.cos(theta),
        np.sin(theta),
        0
    ])

    rirs = {}
    for T60 in T60_values:
        n_samples = int(T60 * fs)

        # Generate RIRs for all microphones
        rir_list = generate(
            c=343,
            fs=fs,
            r=mic_positions,
            s=source_pos,
            L=room_dim,
            reverberation_time=T60,
            nsample=n_samples
        )
        rirs[T60] = rir_list.T

    return rirs


def generate_microphone_signals(
        clean_speech_path,
        fs,
        rirs
):
    # Load clean speech using librosa
    speech, fs_file = librosa.load(
        clean_speech_path,
        sr=fs,
        mono=True
    )

    mic_signals = {}
    for T60, rir_list in rirs.items():
        num_mics = len(rir_list)
        signal_length = len(speech)
        X = np.zeros((num_mics, signal_length), dtype=np.float32)
        for m in range(num_mics):
            # Convolve clean speech with the m RIR
            X[m, :] = fftconvolve(speech, rir_list[m], mode='same')

        mic_signals[T60] = X

    return mic_signals


def generate_white_noise(signal_shape):
    # generate normal distribution samples (mean=0, std=1)
    noise = np.random.randn(*signal_shape)
    return noise


def mix_signals(clean_signal, noise_signal, snr_db):
    # Calculate power of clean signal and noise
    p_signal = np.mean(clean_signal ** 2)
    p_noise = np.mean(noise_signal ** 2)

    # Avoid division by zero
    if p_noise == 0:
        return clean_signal

    # Calculate required scaling factor for noise
    # SNR_linear = P_signal / (scale^2 * P_noise)
    # scale = sqrt(P_signal / (P_noise * 10^(SNR/10)))
    scale_factor = np.sqrt(p_signal / (p_noise * (10 ** (snr_db / 10))))

    # Create noisy signal
    noise = scale_factor * noise_signal
    noisy_signal = clean_signal + noise

    return noisy_signal, noise


def plot_time_freq_analysis(
        original_signal,
        clean_rev_signal,
        noisy_rev_signal,
        fs,
        title_suffix=""
):
    """
    Plots time domain (overlaid) and Spectrograms (STFT) for Original, Clean Reverberant, and Noisy Reverberant signals.
    """
    # 1. Prepare Time Axis
    # Ensure all signals are same length for visualization logic if needed,
    # though usually handled by caller. Taking the minimum length for safe plotting.
    min_len = min(len(original_signal), len(clean_rev_signal), len(noisy_rev_signal))
    original_signal = original_signal[:min_len]
    clean_rev_signal = clean_rev_signal[:min_len]
    noisy_rev_signal = noisy_rev_signal[:min_len]

    t = np.arange(min_len) / fs

    # 2. Setup Figure Grid
    # We will use a grid: Top row for Time Domain (wide), Bottom row for 3 Spectrograms
    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.5], hspace=0.3)

    # --- Time Domain Plot (Top Row) ---
    ax_time = fig.add_subplot(gs[0, :])
    ax_time.plot(t, original_signal, label="Original", alpha=0.8, color='blue', linestyle='--', linewidth=1)
    ax_time.plot(t, clean_rev_signal, label="Clean Reverberant", alpha=0.7, color='green', linewidth=1)
    ax_time.plot(t, noisy_rev_signal, label="Noisy Reverberant", alpha=0.5, color='red', linewidth=1)

    ax_time.set_title(f"Time Domain Signals {title_suffix}")
    ax_time.set_xlabel("Time [s]")
    ax_time.set_ylabel("Amplitude")
    ax_time.legend(loc="upper right")
    ax_time.grid(True, alpha=0.3)
    ax_time.set_xlim([0, t[-1]])

    # --- STFT Helper Function ---
    def plot_spectrogram(signal, ax, title):
        # Compute STFT
        # n_fft: window size, hop_length: step size
        D = librosa.stft(signal, n_fft=512, hop_length=160)
        # Convert to dB (log scale)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # Plot
        img = librosa.display.specshow(S_db, sr=fs, hop_length=160, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
        ax.set_title(title)
        return img

    # --- Spectrograms (Bottom Row) ---
    # 1. Original Dry
    ax_orig = fig.add_subplot(gs[1, 0])
    plot_spectrogram(original_signal, ax_orig, "Original")

    # 2. Clean Reverberant
    ax_clean = fig.add_subplot(gs[1, 1])
    plot_spectrogram(clean_rev_signal, ax_clean, "Clean Reverberant")

    # 3. Noisy Reverberant
    ax_noisy = fig.add_subplot(gs[1, 2])
    img = plot_spectrogram(noisy_rev_signal, ax_noisy, "Noisy Reverberant")

    # Add Colorbar (shared based on the last plot, but represents range roughly for all)
    fig.colorbar(img, ax=[ax_orig, ax_clean, ax_noisy], format='%+2.0f dB', label='Magnitude [dB]')

    plt.suptitle(f"Analysis: {title_suffix}", fontsize=16)
    # plt.tight_layout() # Sometimes conflicts with constrained layouts, used manual spacing above
    plt.show()
