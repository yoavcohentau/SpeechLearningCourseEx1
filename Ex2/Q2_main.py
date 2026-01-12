import os

import numpy as np
from scipy.io import wavfile

from Ex2.Q1_func import generate_room_impulse_responses, generate_microphone_signals, generate_white_noise, mix_signals, \
    plot_time_freq_analysis
from Ex2.Q2_func import apply_dsb

DATA_SET_NAME = "dev-clean"  # "dev-clean" or "test-clean"
DATA_SET_PATH = fr"J:\My Drive\Courses\2026A\Signal Processing and Machine Learning for Speech\HW\HW1\SpeechLearningCourseEx1\data\{DATA_SET_NAME}\LibriSpeech"


def main_q2():
    # Setup Parameters (same as Q1)
    fs = 16000
    room_dim = [4, 5, 3]
    T60 = 0.30
    snr = 10

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

    # --- 2. Generate Signals (Calling Q1 functions) ---
    # NOTE: Assuming generate_room_impulse_responses & generate_microphone_signals exist from Q1
    # We need to recreate the noisy signals for T60=300ms, SNR=10dB

    print("Generating RIRs and Signals...")
    # Target RIR
    target_rirs = generate_room_impulse_responses(fs, room_dim, mic_center, num_mics, mic_spacing, 30, 1.5, [T60])
    target_path = rf'{DATA_SET_PATH}\{DATA_SET_NAME}\84\121123\84-121123-0000.flac'
    target_sigs = generate_microphone_signals(target_path, fs, target_rirs)[T60]

    # Interferer RIR
    inter_rirs = generate_room_impulse_responses(fs, room_dim, mic_center, num_mics, mic_spacing, 150, 2.0, [T60])
    inter_path = rf'{DATA_SET_PATH}\{DATA_SET_NAME}\84\121123\84-121123-0001.flac'
    inter_sigs = generate_microphone_signals(inter_path, fs, inter_rirs)[T60]

    # Cut to same length
    min_len = min(target_sigs.shape[1], inter_sigs.shape[1])
    target_sigs = target_sigs[:, :min_len]
    inter_sigs = inter_sigs[:, :min_len]

    # --- 3. Create Noisy Mixtures ---
    # Case A: White Noise
    white_noise = generate_white_noise(target_sigs.shape)
    noisy_white = mix_signals(target_sigs, white_noise, snr)

    # Case B: Interferer
    noisy_interferer = mix_signals(target_sigs, inter_sigs, snr)

    # --- 4. Apply DSB (Question 2a) ---
    ref_mic_index = 2  # Center mic (0, 1, 2, 3, 4)

    print("Applying Delay-and-Sum Beamformer...")

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

    os.makedirs("output_wavs_q2", exist_ok=True)

    # Save noisy signals
    wavfile.write("output_wavs_q2/white_in.wav", fs, noisy_white[0].astype(np.float32))
    wavfile.write("output_wavs_q2/interferer_in.wav", fs, noisy_interferer[0].astype(np.float32))

    # Plot & Save - White Noise
    print("Plotting White Noise Results...")
    plot_time_freq_analysis(target_clean_ref, noisy_white[0], out_white, fs, "(DSB Output - White Noise)")
    wavfile.write("output_wavs_q2/dsb_white_out.wav", fs, out_white.astype(np.float32))

    # Plot & Save - Interferer
    print("Plotting Interferer Results...")
    plot_time_freq_analysis(target_clean_ref, noisy_interferer[0], out_inter, fs, "(DSB Output - Interferer)")
    wavfile.write("output_wavs_q2/dsb_interferer_out.wav", fs, out_inter.astype(np.float32))

    print("Part A Done.")


if __name__ == "__main__":
    main_q2()
