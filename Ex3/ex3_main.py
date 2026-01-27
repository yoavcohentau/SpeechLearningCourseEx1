from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from Ex3.ex3_func import generate_room_impulse_responses, generate_microphone_signals, apply_srp_phat, \
    generate_white_noise, mix_signals, apply_music, plot_location_maps, calculate_rmse, plot_rmse_performance
from Ex3.librispeech_data_set_utils import LibriSpeechSoundObject


DATA_SET_NAME = "dev-clean"  # "dev-clean" or "test-clean"
DATA_SET_PATH = fr"J:\My Drive\Courses\2026A\Signal Processing and Machine Learning for Speech\HW\HW1\SpeechLearningCourseEx1\data\{DATA_SET_NAME}\LibriSpeech"


def q1_func(source_location, sound, fs, T60_values, snr_values, plot_flag=True):
    # General parameters
    room_dim = [5.2, 6.2, 3.5]

    # Microphone array parameters
    d = 0.2
    mic_locations = [[2.6 - d / 2, 3, 1.5], [2.6 + d / 2, 3, 1.5], [2.6, 3 - d / 2, 1.5], [2.6, 3 + d / 2, 1.5]]
    resolution = [20, 20]

    # Generate target RIRs
    target_rirs = generate_room_impulse_responses(
        fs=fs,
        room_dim=room_dim,
        mic_locations=mic_locations,
        source_location=source_location,
        T60_values=T60_values
    )

    # Plot RIR for the first microphone
    if plot_flag:
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

    target_mic_signals = generate_microphone_signals(
        sound=sound,
        rirs=target_rirs
    )

    if plot_flag:
        mic_index = 0
        num_t60 = len(T60_values)
        fig, axes = plt.subplots(1, num_t60, figsize=(num_t60 * 4, 4), sharey=True, squeeze=False)
        fig.suptitle(f"Reverberant Speech – Microphone {mic_index + 1}", fontsize=16)
        for i, t60_val in enumerate(T60_values):
            signal = target_mic_signals[t60_val][mic_index]
            ax = axes[0, i]
            ax.plot(signal)
            ax.set_title(f"T60 = {int(t60_val * 1000)} ms")
            ax.set_xlabel("Samples")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_ylabel("Amplitude")
        plt.tight_layout()
        plt.show()

    estim_pos_dict = {}
    for t_60 in T60_values:
        for snr in snr_values:
            clean_sig = target_mic_signals[t_60]
            white_noise = generate_white_noise(clean_sig.shape)
            noisy_signal, _ = mix_signals(clean_sig, white_noise, snr)

            srp_map, estimated_pos_srp_phat, (x_range_srp_phat, y_range_srp_phat) = apply_srp_phat(
                mic_sigs=noisy_signal, fs=fs, room_dim=room_dim, mic_locations=mic_locations,
                resolution=resolution, true_source_pos=source_location, plot_map=False)
            music_map, estimated_pos_music, (x_range_music, y_range_music) = apply_music(
                mic_sigs=noisy_signal, fs=fs, room_dim=room_dim, mic_locations=mic_locations,
                resolution=resolution, true_source_pos=source_location, plot_map=False)

            estim_pos_dict[f"{t_60}ms-{snr}db"] = (estimated_pos_srp_phat, estimated_pos_music)

            # Plot maps
            if plot_flag:
                plot_location_maps(
                    maps=[srp_map, music_map],
                    method_names=["SRP-PHAT", "MUSIC"],
                    x_range=x_range_srp_phat,
                    y_range=y_range_srp_phat,
                    true_source_pos=source_location,
                    estimated_positions=[estimated_pos_srp_phat, estimated_pos_music],
                    mic_locations=mic_locations
                )

    return estim_pos_dict


def q2_func(num_of_tries_per_experiment, target_sound, fs, T60_values, snr_values, plot_flag):
    estim_pos_list = []
    source_location_list = []
    for idx_try in tqdm(range(num_of_tries_per_experiment)):
        source_location_list.append(np.array([np.random.uniform(1, 4), np.random.uniform(1, 5), 1.5]))
        estim_pos_list.append(q1_func(source_location=source_location_list[-1],
                                      sound=target_sound,
                                      fs=fs,
                                      T60_values=T60_values,
                                      snr_values=snr_values,
                                      plot_flag=plot_flag))

    result = defaultdict(lambda: [[], []])
    for d in estim_pos_list:
        for key, value in d.items():
            result[key][0].append(value[0])
            result[key][1].append(value[1])
    result = dict(result)

    rmse_results = {}
    for key, (est_list_srp_phat, est_list_music) in result.items():
        rmse_results[key] = {
            "rmse_srp_phat": calculate_rmse(source_location_list, est_list_srp_phat),
            "rmse_music": calculate_rmse(source_location_list, est_list_music),
        }

    return rmse_results


def main_ex3():
    fs = 16000
    T60_values = [0.15, 0.30, 0.55]
    snr_values = [5, 15, 30]  # dB

    # random source locatiom
    source_location = np.array([np.random.uniform(1, 4), np.random.uniform(1, 5), 1.5])

    # load sample
    sound_obj = LibriSpeechSoundObject(
        data_set_path=DATA_SET_PATH,
        data_set_name=DATA_SET_NAME,
        speaker_id='84',
        chapter_number='121123',
        utterance_number='0001',
        file_ext='flac')
    sound, sound_fs = sound_obj.read_file(fs)

    # ---Q1---
    print('---Q1---')

    T60_q1 = T60_values[1]  # 300ms
    snr_q1 = snr_values[1]  # 15dB

    q1_func(source_location, sound, fs, [T60_q1], [snr_q1], True)

    # ---Q2---
    print('---Q2---')
    num_of_tries_per_experiment = 30

    T60_q2_a = [T60_values[1]]  # 300ms
    snr_q2_a = snr_values

    T60_q2_b = T60_values
    snr_q2_b = [snr_values[1]]  # 15dB

    rmse_res_q2_a = q2_func(num_of_tries_per_experiment, sound, fs, T60_q2_a, snr_q2_a, False)
    rmse_res_q2_b = q2_func(num_of_tries_per_experiment, sound, fs, T60_q2_b, snr_q2_b, False)

    plot_rmse_performance([rmse_res_q2_a, rmse_res_q2_b], x_axis_type=["SNR", "T60"])
    pass


if __name__ == "__main__":
    main_ex3()
