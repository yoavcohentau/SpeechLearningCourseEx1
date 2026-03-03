import os

import numpy as np
import torch
from scipy.io import wavfile

from Ex2.Q1_func import generate_room_impulse_responses, generate_microphone_signals, generate_white_noise, mix_signals, \
    plot_time_freq_analysis
from Ex2.Q2_func import apply_mvdr, AudioMetrics, parse_and_plot_results, apply_dsb
from Ex2.Q3_func import load_dns48_model, apply_deep_denoiser
from Ex2.librispeech_data_set_utils import load_librispeech_objects_from_yaml
from Project.Taylor.DeepTaylorBeamformer.nets.TaylorBeamformer import TaylorBeamformer

PLOT_AND_SAVE_FLAG = True
EXAMPLE_IDX_TO_SAVE = 0
SNR_TO_SAVE = 0
T60_TO_SAVE = 0.3
ORIGINAL_SIGNAL_FACTOR = 0.7

DATA_SET_NAME = "dev-clean"  # "dev-clean" or "test-clean"
DATA_SET_PATH = fr"J:\My Drive\Courses\2026A\Signal Processing and Machine Learning for Speech\HW\HW1\SpeechLearningCourseEx1\data\{DATA_SET_NAME}\LibriSpeech"

DNS48_WEIGHTS_PATH = r"C:\Users\Yoav Cohen\Desktop\repositories\SpeechLearningCourseEx1\Ex2\denoiser_weights\dns48-11decc9d8e3f0998.th"


def apply_taylor_net(taylor_net: TaylorBeamformer, sig_in, fs):
    ref_mic = np.mean(sig_in, axis=0)  # ממוצע ערוצים כרפרנס לנרמול
    c = np.sqrt(len(ref_mic) / (np.sum(ref_mic ** 2.0) + 1e-8))
    sig_norm = sig_in * c
    noisy_white = sig_norm

    taylor_net.eval()

    # stft
    win_size = 0.02
    win_shift = 0.01
    fft_num = 320
    device = 'cpu'

    b_size = 1
    channel_num, wav_len = noisy_white.shape
    noisy_white_reshaped = noisy_white.reshape(1, wav_len, channel_num)

    # batch_mix_wav = noisy_white_reshaped.transpose(-2, -1).contiguous().view(b_size * channel_num, wav_len)
    win_size, win_shift = int(fs * win_size), int(fs * win_shift)
    noisy_white_torch = torch.from_numpy(noisy_white).to(device).float()
    batch_mix_stft = torch.stft(
        noisy_white_torch,  # batch_mix_wav,
        n_fft=fft_num,
        hop_length=win_shift,
        win_length=win_size,
        window=torch.hann_window(win_size).to(device),
        return_complex=False)  # (BM,F,T,2)
    batch_frame_list = []

    # for i in range(len(batch_wav_len_list)):
    #     curr_frame_num = (batch_wav_len_list[i] - win_size + win_size) // win_shift + 1  # center case
    #     batch_frame_list.append(curr_frame_num)

    _, freq_num, seq_len, _ = batch_mix_stft.shape
    # batch_mix_stft = batch_mix_stft.view(b_size, -1, freq_num, seq_len, 2)
    batch_mix_stft = batch_mix_stft.reshape(b_size, channel_num, freq_num, seq_len, 2)

    # convert to formats: (B,T,F,M,2) for mix, (B,T,F,2) for target and bf
    batch_mix_stft = batch_mix_stft.permute(0, 3, 2, 1, 4).contiguous()
    # net predict
    with torch.no_grad():
        _, batch_spec_est = taylor_net(batch_mix_stft)  # (B,T,F,2), (B,T,F,2)

    batch_spec_est = batch_spec_est.permute(0, 2, 1, 3).contiguous()
    complex_spec = torch.view_as_complex(batch_spec_est)
    taylor_white_out_torch = torch.istft(
        complex_spec,
        n_fft=fft_num,
        hop_length=win_shift,
        win_length=win_size,
        window=torch.hann_window(win_size).to(device),
        center=True,
        length=wav_len
    )
    taylor_white_out = taylor_white_out_torch.squeeze().cpu().numpy()

    # if len(taylor_white_out) > min_len:
    #     taylor_white_out = taylor_white_out[:min_len]
    # elif len(taylor_white_out) < min_len:
    #     taylor_white_out = np.pad(taylor_white_out, (0, min_len - len(taylor_white_out)))

    return taylor_white_out / c


def align_signal(ref, est):
    corr = np.correlate(est, ref, mode='full')
    shift = np.argmax(corr) - len(ref) + 1

    if shift > 0:
        est = est[shift:]
        ref = ref[:len(est)]
    else:
        ref = ref[-shift:]
        est = est[:len(ref)]

    return ref, est


def taylor_main():
    # Setup Parameters (same as Q1)
    fs = 16000
    room_dim = [4, 5, 3]
    T60_vec = [0.15, 0.3]
    snr_vec = [0, 10]

    mic_center = np.array([2, 1, 1.7])
    num_mics = 5
    mic_spacing = 0.05

    # Mics Positions
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
    all_metrics = []  # Store results for aggregation

    # load samples file names
    yaml_path = '../Ex2/file_name_list.yaml'
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

                # add Noise
                # Case A: White Noise
                white_noise = generate_white_noise(target_sigs.shape)
                noisy_white, white_noise_scaled = mix_signals(target_sigs, white_noise, snr)

                # Case B: Interferer
                noisy_interferer, inter_noise_scaled = mix_signals(target_sigs, inter_sigs, snr)

                ref_mic_index = 2  # Center mic
                ref_noisy_white = noisy_white[ref_mic_index]
                ref_noisy_inter = noisy_interferer[ref_mic_index]
                target_clean_ref = target_sigs[ref_mic_index]

                if PLOT_AND_SAVE_FLAG and example_idx == EXAMPLE_IDX_TO_SAVE and snr == SNR_TO_SAVE and T60 == T60_TO_SAVE:
                    # Save noisy signals
                    wavfile.write("output_folder_proj/white_in.wav", fs, ref_noisy_white.astype(np.float32))
                    wavfile.write("output_folder_proj/interferer_in.wav", fs, ref_noisy_inter.astype(np.float32))


                # # --- (a) DSB ---
                # # ref_mic_index = 2  # Center mic
                #
                # # Apply to White Noise case
                # out_white = apply_dsb(noisy_white, fs, mic_positions, source_pos, ref_mic_index)
                #
                # # Apply to Interferer case
                # out_inter = apply_dsb(noisy_interferer, fs, mic_positions, source_pos, ref_mic_index)
                #
                # # Adjust length
                # out_white = out_white[:min_len]
                # out_inter = out_inter[:min_len]
                #
                # # Plot & Save
                # # For comparison (DSB and MVDR), we look at the Reference Mic (Index 2) of the noisy signal
                # # ref_noisy_white = noisy_white[ref_mic_index]
                # # ref_noisy_inter = noisy_interferer[ref_mic_index]
                # # target_clean_ref = target_sigs[ref_mic_index]
                #
                # # Save metrics
                # metrics[f'DSB-white-{snr}-{T60}-{example_idx}'] = metrics_tool.compute_all(target_clean_ref, out_white)
                # metrics[f'DSB-inter-{snr}-{T60}-{example_idx}'] = metrics_tool.compute_all(target_clean_ref, out_inter)
                #
                # os.makedirs("output_folder_proj", exist_ok=True)
                #
                # if PLOT_AND_SAVE_FLAG and example_idx == 0 and snr == 10 and T60 == 0.3:
                #     # Save noisy signals
                #     wavfile.write("output_folder_proj/white_in.wav", fs, ref_noisy_white.astype(np.float32))
                #     wavfile.write("output_folder_proj/interferer_in.wav", fs, ref_noisy_inter.astype(np.float32))
                #
                #     # White Noise
                #     plot_time_freq_analysis(target_clean_ref/ORIGINAL_SIGNAL_FACTOR, ref_noisy_white, out_white, fs,
                #                             f"(DSB Output - White Noise - T60={T60}s - snr={snr}dB)",
                #                             "Original", "Noisy", "Beamformer Out")
                #     wavfile.write("output_folder_proj/dsb_white_out.wav", fs, out_white.astype(np.float32))
                #
                #     # Interferer
                #     plot_time_freq_analysis(target_clean_ref/ORIGINAL_SIGNAL_FACTOR, ref_noisy_inter, out_inter, fs,
                #                             f"(DSB Output - Interferer - T60={T60}s - snr={snr}dB)",
                #                             "Original", "Noisy", "Beamformer Out")
                #     wavfile.write("output_folder_proj/dsb_interferer_out.wav", fs, out_inter.astype(np.float32))
                #
                # print("Delay-and-Sum Done.")


                # --- (b) MVDR ---
                # Case 1: White Noise
                mvdr_white_out = apply_mvdr(noisy_white, white_noise_scaled)
                mvdr_white_out = mvdr_white_out[:min_len]

                # Case 2: Interferer
                mvdr_inter_out = apply_mvdr(noisy_interferer, inter_noise_scaled)
                mvdr_inter_out = mvdr_inter_out[:min_len]

                # --- Plot & Save ---
                # Save metrics
                # target_clean_ref_1, mvdr_white_out_1 = align_signal(target_clean_ref, mvdr_white_out)
                metrics[f'MVDR-white-{snr}-{T60}-{example_idx}'] = metrics_tool.compute_all(target_clean_ref, mvdr_white_out)
                # target_clean_ref_2, mvdr_inter_out_2 = align_signal(target_clean_ref, mvdr_inter_out)
                metrics[f'MVDR-inter-{snr}-{T60}-{example_idx}'] = metrics_tool.compute_all(target_clean_ref, mvdr_inter_out)

                if PLOT_AND_SAVE_FLAG and example_idx == EXAMPLE_IDX_TO_SAVE and snr == SNR_TO_SAVE and T60 == T60_TO_SAVE:
                # if example_idx == 0 and snr == 10 and T60 == 0.3:
                    # White Noise
                    wavfile.write("output_folder_proj/mvdr_white_out.wav", fs, mvdr_white_out.astype(np.float32))
                    plot_time_freq_analysis(target_clean_ref/ORIGINAL_SIGNAL_FACTOR, ref_noisy_white, mvdr_white_out, fs,
                                            f"(MVDR Output - White Noise - T60={T60}s - snr={snr}dB)",
                                            "Original", "Noisy", "Beamformer Out")

                    # Interferer
                    wavfile.write("output_folder_proj/mvdr_interferer_out.wav", fs, mvdr_inter_out.astype(np.float32))
                    plot_time_freq_analysis(target_clean_ref/ORIGINAL_SIGNAL_FACTOR, ref_noisy_inter, mvdr_inter_out, fs,
                                            f"(MVDR Output - Interferer - T60={T60}s - snr={snr}dB)",
                                            "Original", "Noisy", "Beamformer Out")

                print("MVDR Done.")


                # # --- Q3 - Denoise Net ---
                # # load weights
                # dns_model = load_dns48_model(DNS48_WEIGHTS_PATH)
                #
                # # now the reference is mic_0 as required in Q3
                # first_mic_noisy_white = noisy_white[0]
                # first_mic_noisy_inter = noisy_interferer[0]
                # target_clean_first_mic = target_sigs[0]
                #
                # # Apply denoiser net
                # # White Noise
                # denoiser_white_out = apply_deep_denoiser(first_mic_noisy_white, dns_model)
                # # Interferer
                # denoiser_inter_out = apply_deep_denoiser(first_mic_noisy_inter, dns_model)
                #
                # print("Denoiser Done.")
                #
                # # --- Plot & Save ---
                # # Save metrics
                # metrics[f'Denoiser-white-{snr}-{T60}-{example_idx}'] = metrics_tool.compute_all(target_clean_first_mic, denoiser_white_out)
                # metrics[f'Denoiser-inter-{snr}-{T60}-{example_idx}'] = metrics_tool.compute_all(target_clean_first_mic, denoiser_inter_out)
                #
                # if PLOT_AND_SAVE_FLAG and example_idx == 0 and snr == 10 and T60 == 0.3:
                #     # White Noise
                #     wavfile.write("output_folder_proj/denoiser_white_out.wav", fs, mvdr_white_out.astype(np.float32))
                #     plot_time_freq_analysis(target_clean_ref/ORIGINAL_SIGNAL_FACTOR, ref_noisy_white, denoiser_white_out, fs,
                #                             f"(Denoiser Output - White Noise - T60={T60}s - snr={snr}dB)",
                #                             "Original", "Noisy", "Beamformer Out")
                #
                #     # Interferer
                #     wavfile.write("output_folder_proj/denoiser_interferer_out.wav", fs, mvdr_inter_out.astype(np.float32))
                #     plot_time_freq_analysis(target_clean_ref/ORIGINAL_SIGNAL_FACTOR, ref_noisy_inter, denoiser_inter_out, fs,
                #                             f"(Denoiser Output - Interferer - T60={T60}s - snr={snr}dB)",
                #                             "Original", "Noisy", "Beamformer Out")

                # --- Deep Taylor ---
                # load checkpoints
                checkpoint_load_path = r"J:\My Drive\Courses\YoavAndItayShared\Speech\train_model_folder\BestModels"
                # checkpoint_load_filename = r"best_e28_27_2_26.pth"
                checkpoint_load_filename = r"best_e52_01_3_26.pth"
                checkpoint = torch.load(os.path.join(checkpoint_load_path, checkpoint_load_filename), map_location=torch.device('cpu'))

                taylor_net = TaylorBeamformer(
                            k1=[1, 3],
                            k2=[2, 3],
                            ref_mic=0,
                            c=64,
                            embed_dim=64,
                            fft_num=320,
                            order_num=3,
                            kd1=5,
                            cd1=64,
                            d_feat=256,
                            dilations=[1, 2, 5, 9],
                            group_num=2,
                            hid_node=64,
                            M=5,
                            rnn_type="LSTM",
                            intra_connect="cat",
                            inter_connect="cat",
                            out_type="mapping",
                            bf_type="embedding",
                            norm2d_type="BN",
                            norm1d_type="BN",
                            is_compress=False,
                            is_total_separate=False,
                            is_u2=True,
                            is_1dgate=True,
                            is_squeezed=False,
                            is_causal=True,
                            is_param_share=False
                        )
                # taylor_net.load_state_dict(checkpoint["model_state_dict"])
                taylor_net.load_state_dict(checkpoint)

                # taylor_net.eval()
                #
                # # stft
                # win_size = 0.02
                # win_shift = 0.01
                # fft_num = 320
                # device = 'cpu'
                #
                # b_size = 1
                # channel_num, wav_len = noisy_white.shape
                # noisy_white_reshaped = noisy_white.reshape(1, wav_len, channel_num)
                #
                # # batch_mix_wav = noisy_white_reshaped.transpose(-2, -1).contiguous().view(b_size * channel_num, wav_len)
                # win_size, win_shift = int(fs * win_size), int(fs * win_shift)
                # noisy_white_torch = torch.from_numpy(noisy_white).to(device).float()
                # batch_mix_stft = torch.stft(
                #     noisy_white_torch, #batch_mix_wav,
                #     n_fft=fft_num,
                #     hop_length=win_shift,
                #     win_length=win_size,
                #     window=torch.hann_window(win_size).to(device),
                #     return_complex=False)  # (BM,F,T,2)
                # batch_frame_list = []
                #
                # # for i in range(len(batch_wav_len_list)):
                # #     curr_frame_num = (batch_wav_len_list[i] - win_size + win_size) // win_shift + 1  # center case
                # #     batch_frame_list.append(curr_frame_num)
                #
                # _, freq_num, seq_len, _ = batch_mix_stft.shape
                # # batch_mix_stft = batch_mix_stft.view(b_size, -1, freq_num, seq_len, 2)
                # batch_mix_stft = batch_mix_stft.reshape(1, num_mics, freq_num, seq_len, 2)
                #
                # # convert to formats: (B,T,F,M,2) for mix, (B,T,F,2) for target and bf
                # batch_mix_stft = batch_mix_stft.permute(0, 3, 2, 1, 4).contiguous()
                # # net predict
                # with torch.no_grad():
                #     _, batch_spec_est = taylor_net(batch_mix_stft)  # (B,T,F,2), (B,T,F,2)
                #
                # batch_spec_est = batch_spec_est.permute(0, 2, 1, 3).contiguous()
                # complex_spec = torch.view_as_complex(batch_spec_est)
                # taylor_white_out_torch = torch.istft(
                #     complex_spec,
                #     n_fft=fft_num,
                #     hop_length=win_shift,
                #     win_length=win_size,
                #     window=torch.hann_window(win_size).to(device),
                #     center=True
                # )
                # taylor_white_out = taylor_white_out_torch.squeeze().cpu().numpy()



                # Case 1: White Noise
                taylor_white_out = apply_taylor_net(taylor_net, noisy_white, fs)
                taylor_white_out = taylor_white_out[:min_len]

                # Case 2: Interferer
                taylor_inter_out = apply_taylor_net(taylor_net, noisy_interferer, fs)
                taylor_inter_out = taylor_inter_out[:min_len]

                # --- Plot & Save ---
                # Save metrics
                target_clean_ref_1, taylor_white_out_1 = align_signal(target_clean_ref, taylor_white_out)
                metrics[f'Taylor-white-{snr}-{T60}-{example_idx}'] = metrics_tool.compute_all(target_clean_ref_1,
                                                                                              taylor_white_out_1)
                target_clean_ref_2, taylor_inter_out_2 = align_signal(target_clean_ref, taylor_inter_out)
                metrics[f'Taylor-inter-{snr}-{T60}-{example_idx}'] = metrics_tool.compute_all(target_clean_ref_2,
                                                                                              taylor_inter_out_2)

                if PLOT_AND_SAVE_FLAG and example_idx == EXAMPLE_IDX_TO_SAVE and snr == SNR_TO_SAVE and T60 == T60_TO_SAVE:
                    # if example_idx == 0 and snr == 10 and T60 == 0.3:
                    # White Noise
                    wavfile.write("output_folder_proj/taylor_white_out.wav", fs, taylor_white_out.astype(np.float32))
                    plot_time_freq_analysis(target_clean_ref / ORIGINAL_SIGNAL_FACTOR, ref_noisy_white,
                                            taylor_white_out, fs,
                                            f"(Taylor Output - White Noise - T60={T60}s - snr={snr}dB)",
                                            "Original", "Noisy", "Beamformer Out")

                    # Interferer
                    wavfile.write("output_folder_proj/taylor_interferer_out.wav", fs,
                                  taylor_inter_out.astype(np.float32))
                    plot_time_freq_analysis(target_clean_ref / ORIGINAL_SIGNAL_FACTOR, ref_noisy_inter,
                                            taylor_inter_out, fs,
                                            f"(Taylor Output - Interferer - T60={T60}s - snr={snr}dB)",
                                            "Original", "Noisy", "Beamformer Out")

                print("Taylor Done.")

                # # --- Taylor ---
                # # Case 1: White Noise
                # taylor_white_out = apply_mvdr(noisy_white, white_noise_scaled, use_taylor=True)
                # taylor_white_out = taylor_white_out[:min_len]
                #
                # # Case 2: Interferer
                # taylor_inter_out = apply_mvdr(noisy_interferer, inter_noise_scaled, use_taylor=True)
                # taylor_inter_out = taylor_inter_out[:min_len]
                #
                # # --- Plot & Save ---
                # # Save metrics
                # metrics[f'Taylor-white-{snr}-{T60}-{example_idx}'] = metrics_tool.compute_all(target_clean_ref, taylor_white_out)
                # metrics[f'Taylor-inter-{snr}-{T60}-{example_idx}'] = metrics_tool.compute_all(target_clean_ref, taylor_inter_out)
                #
                # if PLOT_AND_SAVE_FLAG and example_idx == 0 and snr == 10 and T60 == 0.3:
                # # if example_idx == 0 and snr == 10 and T60 == 0.3:
                #     # White Noise
                #     wavfile.write("output_folder_proj/taylor_white_out.wav", fs, taylor_white_out.astype(np.float32))
                #     plot_time_freq_analysis(target_clean_ref/ORIGINAL_SIGNAL_FACTOR, ref_noisy_white, taylor_white_out, fs,
                #                             f"(Taylor Output - White Noise - T60={T60}s - snr={snr}dB)",
                #                             "Original", "Noisy", "Beamformer Out")
                #
                #     # Interferer
                #     wavfile.write("output_folder_proj/taylor_interferer_out.wav", fs, taylor_inter_out.astype(np.float32))
                #     plot_time_freq_analysis(target_clean_ref/ORIGINAL_SIGNAL_FACTOR, ref_noisy_inter, taylor_inter_out, fs,
                #                             f"(Taylor Output - Interferer - T60={T60}s - snr={snr}dB)",
                #                             "Original", "Noisy", "Beamformer Out")
                #
                # print("Taylor Done.")


                all_metrics.append(metrics)

    parse_and_plot_results(all_metrics)


if __name__ == "__main__":
    taylor_main()
