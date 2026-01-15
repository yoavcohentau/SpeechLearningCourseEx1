import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from Ex2.Q1_func import generate_microphone_signals, generate_room_impulse_responses, \
    generate_white_noise, mix_signals, plot_time_freq_analysis
from Ex2.librispeech_data_set_utils import LibriSpeechSoundObject

PLOT_FLAG = True

DATA_SET_NAME = "dev-clean"  # "dev-clean" or "test-clean"
DATA_SET_PATH = fr"J:\My Drive\Courses\2026A\Signal Processing and Machine Learning for Speech\HW\HW1\SpeechLearningCourseEx1\data\{DATA_SET_NAME}\LibriSpeech"


def main_q1():
    # General parameters
    fs = 16000
    room_dim = [4, 5, 3]
    T60_values = [0.15, 0.30]
    snr_values = [0, 10]  # dB

    # Microphone array parameters
    mic_center = [2, 1, 1.7]
    num_mics = 5
    mic_spacing = 0.05

    # Target Source parameters
    source_angle_deg = 30
    source_distance = 1.5

    # Interferer parameters (Noise Type 2)
    interferer_angle_deg = 150
    interferer_distance = 2.0

    # Section (a): Generate target RIRs
    target_rirs = generate_room_impulse_responses(
        fs=fs,
        room_dim=room_dim,
        mic_center=mic_center,
        num_mics=num_mics,
        mic_spacing=mic_spacing,
        source_angle_deg=source_angle_deg,
        source_distance=source_distance,
        T60_values=T60_values
    )

    # Plot time-domain RIR for the first microphone
    if PLOT_FLAG:
        colors = ['b', 'r']
        for i, T60_val in enumerate(T60_values):
            rir_mic1 = target_rirs[T60_val][0]
            rir_mic1 = rir_mic1 / np.max(np.abs(rir_mic1))
            t = np.arange(len(rir_mic1)) / fs
            plt.plot(t, rir_mic1,
                     label=f"T60 = {int(T60_val * 1000)} ms",
                     alpha=0.6,
                     linewidth=0.5,
                     color=colors[i % len(colors)])

        plt.xlabel("Time [s]")
        plt.ylabel("Normalized amplitude")
        plt.title("Room Impulse Responses – First Microphone")
        plt.legend()
        plt.grid()
        plt.show()

    # Section (b)
    target_sound_obj = LibriSpeechSoundObject(
        data_set_path=DATA_SET_PATH,
        data_set_name=DATA_SET_NAME,
        speaker_id='84',
        chapter_number='121123',
        utterance_number='0000',
        file_ext='flac')
    target_file_path = target_sound_obj.params2path()

    target_mic_signals = generate_microphone_signals(
        clean_speech_path=target_file_path,
        fs=fs,
        rirs=target_rirs
    )

    if PLOT_FLAG:
        T60_test = T60_values[1]
        mic_index = 0

        plt.plot(target_mic_signals[T60_test][mic_index])
        plt.title("Reverberant speech – Microphone 1")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()

    # Section (c)
    interferer_rirs = generate_room_impulse_responses(
        fs=fs,
        room_dim=room_dim,
        mic_center=mic_center,
        num_mics=num_mics,
        mic_spacing=mic_spacing,
        source_angle_deg=interferer_angle_deg,
        source_distance=interferer_distance,
        T60_values=T60_values
    )

    interferer_sound_obj = LibriSpeechSoundObject(
        data_set_path=DATA_SET_PATH,
        data_set_name=DATA_SET_NAME,
        speaker_id='84',
        chapter_number='121123',
        utterance_number='0001',
        file_ext='flac')
    interferer_file_path = interferer_sound_obj.params2path()

    interferer_mic_signals = generate_microphone_signals(
        clean_speech_path=interferer_file_path,
        fs=fs,
        rirs=interferer_rirs)

    original_target_sound, _ = target_sound_obj.read_file(fs=fs)

    # Create output directory for wav files
    output_dir = "output_folder_q1"
    os.makedirs(output_dir, exist_ok=True)

    # We focus on T60 = 300ms (0.3s) and SNR = 10dB for the specific plots requested in section (d)
    plot_T60 = T60_values[1]
    plot_SNR = snr_values[1]

    for T60 in T60_values:
        for snr in snr_values:

            # Get Clean Target Signal (Multichannel)
            clean_sig = target_mic_signals[T60]

            # Type 1 - White Gaussian Noise
            white_noise = generate_white_noise(clean_sig.shape)
            noisy_white, _ = mix_signals(clean_sig, white_noise, snr)

            # Type 2 - Interferer Noise
            inter_sig = interferer_mic_signals[T60]

            # Ensure lengths match (Interferer might be longer/shorter)
            min_len = min(clean_sig.shape[1], inter_sig.shape[1])
            clean_sig_trunc = clean_sig[:, :min_len]
            inter_sig_trunc = inter_sig[:, :min_len]

            noisy_interferer, _ = mix_signals(clean_sig_trunc, inter_sig_trunc, snr)

            # Section (c) - plot
            if T60 == plot_T60 and snr == plot_SNR:  # Only for T60=300ms and SNR=10dB
                # 1 - White Noise (mic 1)
                plot_time_freq_analysis(
                    original_target_sound[:clean_sig.shape[1]],
                    clean_sig[0, :],
                    noisy_white[0, :],
                    fs,
                    title_suffix=f"(White Noise, SNR={snr}dB, T60={T60}s)"
                )

                # 2 - Interferer (mic 1)
                plot_time_freq_analysis(
                    original_target_sound[:min_len],
                    clean_sig_trunc[0, :],
                    noisy_interferer[0, :],
                    fs,
                    title_suffix=f"(Interferer, SNR={snr}dB, T60={T60}s)"
                )

            # Section (e) - Save WAV files (mic 1)
            # - save clean reverberation version
            clean_filename_suffix = f"T60_{int(T60 * 1000)}ms"
            if snr == snr_values[0]:
                wavfile.write(
                    f"{output_dir}/clean_{clean_filename_suffix}.wav",
                    fs,
                    clean_sig[0, :].astype(np.float32)
                )

            filename_suffix = f"T60_{int(T60 * 1000)}ms_SNR_{snr}dB"

            # - save White Noise version
            wavfile.write(
                f"{output_dir}/white_noise_{filename_suffix}.wav",
                fs,
                noisy_white[0, :].astype(np.float32)
            )

            # - save Interferer version
            wavfile.write(
                f"{output_dir}/interferer_{filename_suffix}.wav",
                fs,
                noisy_interferer[0, :].astype(np.float32)
            )


if __name__ == "__main__":
    main_q1()
