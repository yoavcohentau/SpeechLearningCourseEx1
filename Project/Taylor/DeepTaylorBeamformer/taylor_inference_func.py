import numpy as np
import torch

from Project.Taylor.DeepTaylorBeamformer.nets.TaylorBeamformer import TaylorBeamformer


def apply_taylor_net(taylor_net: TaylorBeamformer, sig_in, fs):
    ref_mic = np.mean(sig_in, axis=0)
    c = np.sqrt(len(ref_mic) / (np.sum(ref_mic ** 2.0) + 1e-8))
    sig_norm = sig_in * c
    noisy_sig = sig_norm

    taylor_net.eval()

    # stft
    win_size = 0.02
    win_shift = 0.01
    fft_num = 320
    device = 'cpu'

    b_size = 1
    channel_num, wav_len = noisy_sig.shape

    win_size, win_shift = int(fs * win_size), int(fs * win_shift)
    noisy_sig_torch = torch.from_numpy(noisy_sig).to(device).float()
    batch_mix_stft = torch.stft(
        noisy_sig_torch,  # batch_mix_wav,
        n_fft=fft_num,
        hop_length=win_shift,
        win_length=win_size,
        window=torch.hann_window(win_size).to(device),
        return_complex=False)  # (BM,F,T,2)

    _, freq_num, seq_len, _ = batch_mix_stft.shape
    batch_mix_stft = batch_mix_stft.reshape(b_size, channel_num, freq_num, seq_len, 2)

    # convert to formats: (B,T,F,M,2) for mix, (B,T,F,2) for target and bf
    batch_mix_stft = batch_mix_stft.permute(0, 3, 2, 1, 4).contiguous()
    # net predict
    with torch.no_grad():
        _, batch_spec_est = taylor_net(batch_mix_stft)  # (B,T,F,2), (B,T,F,2)

    batch_spec_est = batch_spec_est.permute(0, 2, 1, 3).contiguous()
    complex_spec = torch.view_as_complex(batch_spec_est)
    taylor_sig_out_torch = torch.istft(
        complex_spec,
        n_fft=fft_num,
        hop_length=win_shift,
        win_length=win_size,
        window=torch.hann_window(win_size).to(device),
        center=True,
        length=wav_len
    )
    taylor_sig_out = taylor_sig_out_torch.squeeze().cpu().numpy()

    return taylor_sig_out / c


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
