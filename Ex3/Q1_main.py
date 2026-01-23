import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from Ex3.Q1_func import generate_room_impulse_responses, generate_microphone_signals, apply_srp_phat
from Ex3.librispeech_data_set_utils import LibriSpeechSoundObject

PLOT_FLAG = False
ORIGINAL_SIGNAL_FACTOR = 5

DATA_SET_NAME = "dev-clean"  # "dev-clean" or "test-clean"
DATA_SET_PATH = fr"J:\My Drive\Courses\2026A\Signal Processing and Machine Learning for Speech\HW\HW1\SpeechLearningCourseEx1\data\{DATA_SET_NAME}\LibriSpeech"


def main_q1():
    # General parameters
    fs = 16000
    room_dim = [5.2, 6.2, 3.5]
    T60_values = [0.15, 0.30, 0.55]
    snr_values = [5, 15, 30]  # dB

    # Microphone array parameters
    d = 0.2
    mic_locations = [[2.6-d/2, 3, 1.5], [2.6+d/2, 3, 1.5], [2.6, 3-d/2, 1.5], [2.6, 3+d/2, 1.5]]
    num_mics = len(mic_locations)
    mic_spacing = 0.05

    # Target Source parameters
    source_location = np.array([np.random.uniform(1, 4), np.random.uniform(1, 5), 1.5])

    # Generate target RIRs
    target_rirs = generate_room_impulse_responses(
        fs=fs,
        room_dim=room_dim,
        mic_locations=mic_locations,
        source_location=source_location,
        T60_values=T60_values
    )

    # Plot time-domain RIR for the first microphone
    if PLOT_FLAG:
        colors = ['b', 'r', 'g']
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

    # load sample path
    target_sound_obj = LibriSpeechSoundObject(
        data_set_path=DATA_SET_PATH,
        data_set_name=DATA_SET_NAME,
        speaker_id='84',
        chapter_number='121123',
        utterance_number='0000',
        file_ext='flac')
    target_sound, target_sound_fs = target_sound_obj.read_file(fs)

    target_mic_signals = generate_microphone_signals(
        sound=target_sound,
        rirs=target_rirs
    )

    if PLOT_FLAG:
        mic_index = 0
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        fig.suptitle(f"Reverberant Speech – Microphone {mic_index + 1}", fontsize=16)
        for i, t60_val in enumerate(T60_values):
            signal = target_mic_signals[t60_val][mic_index]
            axes[i].plot(signal)
            axes[i].set_title(f"T60 = {int(t60_val * 1000)} ms")
            axes[i].set_xlabel("Samples")
            axes[i].grid(True, alpha=0.3)
            if i == 0:
                axes[i].set_ylabel("Amplitude")
        plt.tight_layout()
        plt.show()

    # ---Q1---
    T60_q1 = T60_values[1]
    snr_q1 = snr_values[1]

    loc_map = apply_srp_phat(mic_sigs=target_mic_signals[T60_q1], fs=fs, room_dim=room_dim, mic_locations=mic_locations, resolution=[200, 200], true_source_pos=source_location)




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
                    original_target_sound[:clean_sig.shape[1]]/ORIGINAL_SIGNAL_FACTOR,
                    clean_sig[0, :],
                    noisy_white[0, :],
                    fs,
                    title_suffix=f"(White Noise, SNR={snr}dB, T60={T60}s)"
                )

                # 2 - Interferer (mic 1)
                plot_time_freq_analysis(
                    original_target_sound[:min_len]/ORIGINAL_SIGNAL_FACTOR,
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
