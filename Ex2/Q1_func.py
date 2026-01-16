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
            # X[m, :] = fftconvolve(speech, rir_list[m], mode='same')
            x_fft = fftconvolve(speech, rir_list[m])
            X[m, :] = x_fft[:signal_length]

        mic_signals[T60] = X

    return mic_signals


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


def generate_white_noise(signal_shape):
    # generate normal distribution samples (mean=0, std=1)
    noise = np.random.randn(*signal_shape)
    return noise


def plot_time_freq_analysis(
        signal_1,
        signal_2,
        signal_3,
        fs=16000,
        title_suffix="",
        legend_1='Original',
        legend_2='Clean Reverberant',
        legend_3='Noisy Reverberant'
):
    min_len = min(len(signal_1), len(signal_2), len(signal_3))
    signal_1 = signal_1[:min_len]
    signal_2 = signal_2[:min_len]
    signal_3 = signal_3[:min_len]

    t = np.arange(min_len) / fs

    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.5], hspace=0.3)

    # --- Time Domain Plot (Top Row) ---
    ax_time = fig.add_subplot(gs[0, :])
    ax_time.plot(t, signal_1, label=legend_1, alpha=0.8, color='blue', linestyle='--', linewidth=1)
    ax_time.plot(t, signal_2, label=legend_2, alpha=0.7, color='green', linewidth=1)
    ax_time.plot(t, signal_3, label=legend_3, alpha=0.5, color='red', linewidth=1)

    ax_time.set_title(f"Time Domain Signals {title_suffix}")
    ax_time.set_xlabel("Time [s]")
    ax_time.set_ylabel("Amplitude")
    ax_time.legend(loc="upper right")
    ax_time.grid(True, alpha=0.3)
    ax_time.set_xlim([0, t[-1]])

    # --- STFT Helper Function ---
    def plot_spectrogram(signal, ax, title):
        # Compute STFT
        D = librosa.stft(signal, n_fft=512, hop_length=160)
        # Convert to dB
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        # Plot
        img = librosa.display.specshow(S_db, sr=fs, hop_length=160, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
        ax.set_title(title)
        return img

    # --- Spectrograms Plot (Bottom Row) ---
    # signal_1
    ax_orig = fig.add_subplot(gs[1, 0])
    plot_spectrogram(signal_1, ax_orig, legend_1)

    # signal_2
    ax_clean = fig.add_subplot(gs[1, 1])
    plot_spectrogram(signal_2, ax_clean, legend_2)

    # signal_3
    ax_noisy = fig.add_subplot(gs[1, 2])
    img = plot_spectrogram(signal_3, ax_noisy, legend_3)

    # Add Colorbar
    fig.colorbar(img, ax=[ax_orig, ax_clean, ax_noisy], format='%+2.0f dB', label='Magnitude [dB]')

    plt.suptitle(f"Analysis: {title_suffix}", fontsize=16)
    # plt.tight_layout() # Sometimes conflicts with constrained layouts, used manual spacing above
    plt.show()

