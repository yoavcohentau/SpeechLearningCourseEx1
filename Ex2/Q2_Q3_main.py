import os

import numpy as np
from scipy.io import wavfile

from Ex2.Q1_func import generate_room_impulse_responses, generate_microphone_signals, generate_white_noise, mix_signals, \
    plot_time_freq_analysis
from Ex2.Q2_func import apply_dsb, apply_mvdr, AudioMetrics, parse_and_plot_results
from Ex2.Q3_func import load_dns48_model, apply_deep_denoiser
from Ex2.librispeech_data_set_utils import load_librispeech_objects_from_yaml

PLOT_AND_SAVE_FLAG = True

DATA_SET_NAME = "dev-clean"  # "dev-clean" or "test-clean"
DATA_SET_PATH = fr"J:\My Drive\Courses\2026A\Signal Processing and Machine Learning for Speech\HW\HW1\SpeechLearningCourseEx1\data\{DATA_SET_NAME}\LibriSpeech"

DNS48_WEIGHTS_PATH = r"C:\Users\Yoav Cohen\Desktop\repositories\SpeechLearningCourseEx1\Ex2\denoiser_weights\dns48-11decc9d8e3f0998.th"


def main_q2():
    # Setup Parameters (same as Q1)
    fs = 16000
    room_dim = [4, 5, 3]
    T60_vec = [0.15, 0.3]
    snr_vec = [0, 10]

    mic_center = np.array([2, 1, 1.7])
    num_mics = 5
    mic_spacing = 0.05

    # Define Mic Positions (needed for Q2)
    mic_positions = np.array([
        mic_center + np.array([(i - (num_mics - 1) / 2) * mic_spacing, 0, 0])
        for i in range(num_mics)
    ])

    # Source Parameters
    # Target
    src_theta = np.deg2rad(30)
    source_pos = mic_center + 1.5 * np.array([np.cos(src_theta), np.sin(src_theta), 0])

    # Interferer (for noise type 2)
    int_theta = np.deg2rad(150)
    interferer_pos = mic_center + 2.0 * np.array([np.cos(int_theta), np.sin(int_theta), 0])

    # Initialize Metrics Calculator
    metrics_tool = AudioMetrics(fs)

    # Store results for aggregation
    all_metrics = []

    # --- 2. Generate Signals (Calling Q1 functions) ---
    # NOTE: Assuming generate_room_impulse_responses & generate_microphone_signals exist from Q1
    # We need to recreate the noisy signals for T60=300ms, SNR=10dB

    yaml_path = 'file_name_list.yaml'
    signal_objects, interferer_objects = load_librispeech_objects_from_yaml(
        yaml_path,
        DATA_SET_PATH,
        DATA_SET_NAME
    )
    for T60 in T60_vec:
        for snr in snr_vec:
            for example_idx, (signal_object, interferer_object) in enumerate(zip(signal_objects, interferer_objects)):
                print(f'---------- example #{example_idx} ----------')

                metrics = {}

                # Target RIR
                target_rirs = generate_room_impulse_responses(fs, room_dim, mic_center, num_mics, mic_spacing, 30, 1.5, [T60])
                target_path = signal_object.params2path()
                target_sigs = generate_microphone_signals(target_path, fs, target_rirs)[T60]

                # Interferer RIR
                inter_rirs = generate_room_impulse_responses(fs, room_dim, mic_center, num_mics, mic_spacing, 150, 2.0, [T60])
                inter_path = interferer_object.params2path()
                inter_sigs = generate_microphone_signals(inter_path, fs, inter_rirs)[T60]

                # Cut to same length
                min_len = min(target_sigs.shape[1], inter_sigs.shape[1])
                target_sigs = target_sigs[:, :min_len]
                inter_sigs = inter_sigs[:, :min_len]

                # --- 3. Create Noisy Mixtures ---
                # Case A: White Noise
                white_noise = generate_white_noise(target_sigs.shape)
                noisy_white, white_noise_scaled = mix_signals(target_sigs, white_noise, snr)

                # Case B: Interferer
                noisy_interferer, inter_noise_scaled = mix_signals(target_sigs, inter_sigs, snr)

                # --- 4. Apply DSB (Question 2a) ---
                ref_mic_index = 2  # Center mic (0, 1, 2, 3, 4)

                # Apply to White Noise case
                out_white = apply_dsb(noisy_white, fs, mic_positions, source_pos, ref_mic_index)

                # Apply to Interferer case
                out_inter = apply_dsb(noisy_interferer, fs, mic_positions, source_pos, ref_mic_index)

                # Adjust length (ISTFT might add a few samples)
                out_white = out_white[:min_len]
                out_inter = out_inter[:min_len]

                # --- 5. Visualization & Saving ---
                # For comparison, we look at the Reference Mic (Index 2) of the noisy signal
                ref_noisy_white = noisy_white[ref_mic_index]
                ref_noisy_inter = noisy_interferer[ref_mic_index]
                target_clean_ref = target_sigs[ref_mic_index]

                # Save metrics
                metrics[f'DSB-white-{snr}-{T60}-{example_idx}'] = metrics_tool.compute_all(target_clean_ref, out_white)
                metrics[f'DSB-inter-{snr}-{T60}-{example_idx}'] = metrics_tool.compute_all(target_clean_ref, out_inter)

                os.makedirs("output_folder_q2", exist_ok=True)

                if PLOT_AND_SAVE_FLAG and example_idx == 0 and snr == 10 and T60 == 0.3:
                    # Save noisy signals
                    wavfile.write("output_folder_q2/white_in.wav", fs, ref_noisy_white.astype(np.float32))
                    wavfile.write("output_folder_q2/interferer_in.wav", fs, ref_noisy_inter.astype(np.float32))

                    # Plot & Save - White Noise
                    plot_time_freq_analysis(target_clean_ref, ref_noisy_white, out_white, fs,
                                            f"(DSB Output - White Noise - T60={T60}s - snr={snr}dB)",
                                            "Original", "Noisy", "Beamformer Out")
                    wavfile.write("output_folder_q2/dsb_white_out.wav", fs, out_white.astype(np.float32))

                    # Plot & Save - Interferer
                    plot_time_freq_analysis(target_clean_ref, ref_noisy_inter, out_inter, fs,
                                            f"(DSB Output - Interferer - T60={T60}s - snr={snr}dB)",
                                            "Original", "Noisy", "Beamformer Out")
                    wavfile.write("output_folder_q2/dsb_interferer_out.wav", fs, out_inter.astype(np.float32))

                print("Delay-and-Sum Done.")


                # --- Part B: MVDR Beamformer ---
                # Case 1: White Noise
                mvdr_white_out = apply_mvdr(noisy_white, white_noise_scaled)
                # Trim
                mvdr_white_out = mvdr_white_out[:min_len]

                # Case 2: Interferer
                mvdr_inter_out = apply_mvdr(noisy_interferer, inter_noise_scaled)
                # Trim
                mvdr_inter_out = mvdr_inter_out[:min_len]

                # --- Visualization & Saving ---
                # Save metrics
                metrics[f'MVDR-white-{snr}-{T60}-{example_idx}'] = metrics_tool.compute_all(target_clean_ref, mvdr_white_out)
                metrics[f'MVDR-inter-{snr}-{T60}-{example_idx}'] = metrics_tool.compute_all(target_clean_ref, mvdr_inter_out)

                if PLOT_AND_SAVE_FLAG and example_idx == 0 and snr == 10 and T60 == 0.3:
                    # White Noise
                    wavfile.write("output_folder_q2/mvdr_white_out.wav", fs, mvdr_white_out.astype(np.float32))
                    plot_time_freq_analysis(target_clean_ref, ref_noisy_white, mvdr_white_out, fs,
                                            f"(MVDR Output - White Noise - T60={T60}s - snr={snr}dB)",
                                            "Original", "Noisy", "Beamformer Out")

                    # Interferer
                    wavfile.write("output_folder_q2/mvdr_interferer_out.wav", fs, mvdr_inter_out.astype(np.float32))
                    plot_time_freq_analysis(target_clean_ref, ref_noisy_inter, mvdr_inter_out, fs,
                                            f"(MVDR Output - Interferer - T60={T60}s - snr={snr}dB)",
                                            "Original", "Noisy", "Beamformer Out")

                print("MVDR Done.")


                # --- Q3: Denoise Net ---
                dns_model = load_dns48_model(DNS48_WEIGHTS_PATH)

                first_mic_noisy_white = noisy_white[0]
                first_mic_noisy_inter = noisy_interferer[0]
                target_clean_first_mic = target_sigs[0]
                denoiser_white_out = apply_deep_denoiser(first_mic_noisy_white, dns_model, fs)
                denoiser_inter_out = apply_deep_denoiser(first_mic_noisy_inter, dns_model, fs)
                print("Denoiser Done.")

                # Save metrics
                metrics[f'Denoiser-white-{snr}-{T60}-{example_idx}'] = metrics_tool.compute_all(target_clean_first_mic, denoiser_white_out)
                metrics[f'Denoiser-inter-{snr}-{T60}-{example_idx}'] = metrics_tool.compute_all(target_clean_first_mic, denoiser_inter_out)

                if PLOT_AND_SAVE_FLAG and example_idx == 0 and snr == 10 and T60 == 0.3:
                    # White Noise
                    wavfile.write("output_folder_q2/denoiser_white_out.wav", fs, mvdr_white_out.astype(np.float32))
                    plot_time_freq_analysis(target_clean_ref, ref_noisy_white, denoiser_white_out, fs,
                                            f"(Denoiser Output - White Noise - T60={T60}s - snr={snr}dB)",
                                            "Original", "Noisy", "Beamformer Out")

                    # Interferer
                    wavfile.write("output_folder_q2/denoiser_interferer_out.wav", fs, mvdr_inter_out.astype(np.float32))
                    plot_time_freq_analysis(target_clean_ref, ref_noisy_inter, denoiser_inter_out, fs,
                                            f"(Denoiser Output - Interferer - T60={T60}s - snr={snr}dB)",
                                            "Original", "Noisy", "Beamformer Out")

                all_metrics.append(metrics)

                pass
            pass
        pass
    parse_and_plot_results(all_metrics)
    pass


if __name__ == "__main__":
    main_q2()
