import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from Ex3.ex3_func import generate_room_impulse_responses, generate_microphone_signals, apply_srp_phat, \
    generate_white_noise, mix_signals, apply_music, plot_location_maps
from Ex3.librispeech_data_set_utils import LibriSpeechSoundObject

PLOT_FLAG = False
ORIGINAL_SIGNAL_FACTOR = 5

DATA_SET_NAME = "dev-clean"  # "dev-clean" or "test-clean"
DATA_SET_PATH = fr"J:\My Drive\Courses\2026A\Signal Processing and Machine Learning for Speech\HW\HW1\SpeechLearningCourseEx1\data\{DATA_SET_NAME}\LibriSpeech"


def main_q1():
def main_ex3():
    # General parameters
    fs = 16000
    room_dim = [5.2, 6.2, 3.5]
    T60_values = [0.15, 0.30, 0.55]
    snr_values = [5, 15, 30]  # dB

    # Microphone array parameters
    d = 0.2
    mic_locations = [[2.6 - d / 2, 3, 1.5], [2.6 + d / 2, 3, 1.5], [2.6, 3 - d / 2, 1.5], [2.6, 3 + d / 2, 1.5]]

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

    clean_sig = target_mic_signals[T60_q1]
    white_noise = generate_white_noise(clean_sig.shape)
    noisy_signal, _ = mix_signals(clean_sig, white_noise, snr_q1)

    srp_map, estimated_pos_srp_phat, (x_range_srp_phat, y_range_srp_phat) = apply_srp_phat(
        mic_sigs=noisy_signal, fs=fs, room_dim=room_dim, mic_locations=mic_locations,
        resolution=[20, 20], true_source_pos=source_location, plot_map=False)
    music_map, estimated_pos_music, (x_range_music, y_range_music) = apply_music(
        mic_sigs=noisy_signal, fs=fs, room_dim=room_dim, mic_locations=mic_locations,
        resolution=[20, 20], true_source_pos=source_location, plot_map=False)

    # Plot maps
    plot_location_maps(
        maps=[srp_map, music_map],
        method_names=["SRP-PHAT", "MUSIC"],
        x_range=x_range_srp_phat,
        y_range=y_range_srp_phat,
        true_source_pos=source_location,
        estimated_positions=[estimated_pos_srp_phat, estimated_pos_music],
        mic_locations=mic_locations
    )

    # ---Q2---



if __name__ == "__main__":
    main_ex3()
