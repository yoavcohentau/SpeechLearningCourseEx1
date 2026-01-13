import numpy as np
from scipy.signal import stft, istft


# ---------- DSB ----------

def compute_steering_vector(mic_positions, source_pos, freqs, ref_mic_index=None, c=343):
    I = len(mic_positions)

    # mics positions relative to the reference mic
    q = np.array([np.linalg.norm(mic_pos - source_pos) for mic_pos in mic_positions])
    if ref_mic_index is not None:
        q = q - q[ref_mic_index]

    # Compute d(f)
    num_freqs = len(freqs)
    d = np.zeros((num_freqs, I), dtype=np.complex64)
    for i in range(I):
        d[:, i] = np.exp(-2j * np.pi * freqs * q[i] / c)

    return d


def apply_dsb(mic_signals, fs, mic_positions, source_pos, ref_mic_index):
    # params
    num_mics = mic_signals.shape[0]
    nperseg = 512
    noverlap = 256

    # STFT for each channel
    stft_list = []  # (num_mics, f, t)
    for m in range(num_mics):
        f, t, stft_vals = stft(mic_signals[m], fs=fs, nperseg=nperseg, noverlap=noverlap)
        stft_list.append(stft_vals)

    x = np.stack(stft_list, axis=-1)  # transpose to (f, t, mics)

    # steering vector
    d = compute_steering_vector(mic_positions, source_pos, f, ref_mic_index)

    # DSB weights
    w = d / num_mics
    w_conj = np.conj(w.reshape(len(f), 1, num_mics))

    # apply beamformer: z(t, f) = w^H(f) * x(t, f)
    z = np.sum(w_conj * x, axis=2)  # (freqs, time)

    # ISTFT
    _, out_signal = istft(z, fs=fs, nperseg=nperseg, noverlap=noverlap)

    return out_signal


# ---------- MVDR ----------

import numpy as np
from scipy.linalg import eigh, inv
from scipy.signal import stft, istft


# --- Step 1: Estimate Noise Covariance ---
def estimate_noise_covariance(noise_signals, fs=16000, nperseg=512, noverlap=256):
    """
    Estimates the noise covariance matrix (Phi_vv) based on noise signals.
    Instruction B.1
    """
    num_mics = noise_signals.shape[0]

    # Calculate STFT for noise signals
    stft_list = []
    for m in range(num_mics):
        f, t, Zxx = stft(noise_signals[m], fs=fs, nperseg=nperseg, noverlap=noverlap)
        stft_list.append(Zxx)

    # Stack to (Freqs, Frames, Mics)
    N_stft = np.stack(stft_list, axis=-1)

    num_freqs = N_stft.shape[0]
    num_frames = N_stft.shape[1]

    # Initialize Covariance Matrix: (Freqs, Mics, Mics)
    Phi_vv = np.zeros((num_freqs, num_mics, num_mics), dtype=np.complex64)

    for f in range(num_freqs):
        N_f = N_stft[f, :, :]  # (Frames, Mics)
        # E[n * n^H] -> Average over frames
        R = np.matmul(N_f.conj().T, N_f) / num_frames
        # Ensure Hermitian symmetry
        Phi_vv[f] = (R + R.conj().T) / 2

    return Phi_vv


# --- Step 2: Estimate RTF using GEVD ---
def estimate_rtf_gevd(noisy_signals, Phi_vv, ref_mic_index=2, fs=16000, nperseg=512, noverlap=256):
    """
    Estimates RTF using GEVD based on noisy speech signals and estimated noise covariance.
    Instruction B.2

    Method:
    1. Calculate Noisy Covariance (Phi_xx).
    2. Solve GEVD: Phi_xx * v = lambda * Phi_vv * v
    3. RTF d is proportional to Phi_vv * v.
    """
    num_mics = noisy_signals.shape[0]

    # 1. Calculate STFT for noisy signals
    stft_list = []
    for m in range(num_mics):
        _, _, Zxx = stft(noisy_signals[m], fs=fs, nperseg=nperseg, noverlap=noverlap)
        stft_list.append(Zxx)
    X_stft = np.stack(stft_list, axis=-1)

    num_freqs = Phi_vv.shape[0]
    num_frames = X_stft.shape[1]

    d_hat = np.zeros((num_freqs, num_mics), dtype=np.complex64)

    # Regularization factor (Diagonal Loading) is CRITICAL here.
    # Without it, Phi_vv is singular for single interferer -> GEVD fails -> Stripes.
    reg_epsilon = 1e-3

    for f in range(num_freqs):
        # Estimate Phi_xx
        X_f = X_stft[f, :, :]
        R_xx = np.matmul(X_f.conj().T, X_f) / num_frames
        R_xx = (R_xx + R_xx.conj().T) / 2

        # Regularize Phi_vv for GEVD stability
        # We assume Phi_vv is provided from Step 1
        trace_vv = np.trace(Phi_vv[f]).real
        loading = reg_epsilon * trace_vv + 1e-6
        R_vv_reg = Phi_vv[f] + loading * np.eye(num_mics)

        try:
            # Solve Generalized Eigenvalue Problem
            # scipy.linalg.eigh(a, b) solves a*v = w*b*v
            vals, vecs = eigh(R_xx, R_vv_reg)

            # The eigenvector corresponding to the LARGEST eigenvalue
            v = vecs[:, -1]

            # The RTF d is proportional to Phi_vv * v
            # Reference: Gannot et al., "Consolidated Perspective...", 2017 [cite: 683]
            d_unnorm = np.matmul(R_vv_reg, v)

            # Normalize by reference microphone
            ref_val = d_unnorm[ref_mic_index]

            # Safety check for numerical zeros (avoiding "stripes" in silent bins)
            if np.abs(ref_val) < 1e-6:
                d_hat[f] = np.ones(num_mics)
            else:
                d_hat[f] = d_unnorm / ref_val

        except np.linalg.LinAlgError:
            d_hat[f] = np.ones(num_mics)

    return d_hat


# --- Step 3: Compute MVDR Weights ---
def compute_mvdr_weights(Phi_vv, d_rtf):
    """
    Calculates the filter weights.
    Instruction B.3
    w = (Phi_vv^-1 * d) / (d^H * Phi_vv^-1 * d)
    """
    num_freqs, num_mics, _ = Phi_vv.shape
    w_mvdr = np.zeros((num_freqs, num_mics), dtype=np.complex64)

    reg_epsilon = 1e-3

    for f in range(num_freqs):
        # Regularize for Inversion
        trace_vv = np.trace(Phi_vv[f]).real
        loading = reg_epsilon * trace_vv + 1e-6
        R_inv = inv(Phi_vv[f] + loading * np.eye(num_mics))

        d = d_rtf[f, :, np.newaxis]  # Column vector

        # MVDR formula [cite: 322]
        num = np.matmul(R_inv, d)
        denom = np.matmul(d.conj().T, num).item()

        if np.abs(denom) < 1e-12:
            w_mvdr[f] = np.ones(num_mics) / num_mics
        else:
            w_mvdr[f] = (num / denom).flatten()

    return w_mvdr


# --- Apply Function (Orchestrator) ---
def apply_mvdr(noisy_signals, noise_signals_only, fs=16000, ref_mic_index=2):
    """
    Full pipeline orchestrator calling steps 1, 2, 3.
    """
    # 1. Estimate Noise Covariance (Phi_vv)
    Phi_vv = estimate_noise_covariance(noise_signals_only, fs=fs)

    # 2. Estimate RTF using GEVD (based on Noisy Signals and Phi_vv)
    d_rtf = estimate_rtf_gevd(noisy_signals, Phi_vv, ref_mic_index=ref_mic_index, fs=fs)

    # 3. Compute Weights
    w_mvdr = compute_mvdr_weights(Phi_vv, d_rtf)

    # --- Apply to Signal ---
    # Need STFT of noisy signal again to apply weights
    stft_list = []
    for m in range(noisy_signals.shape[0]):
        _, _, Zxx = stft(noisy_signals[m], fs=fs, nperseg=512, noverlap=256)
        stft_list.append(Zxx)
    X_stft = np.stack(stft_list, axis=-1)

    # w^H * x
    w_conj = np.conj(w_mvdr[:, np.newaxis, :])
    Z_out = np.sum(w_conj * X_stft, axis=2)

    # ISTFT
    _, out_time = istft(Z_out, fs=fs, nperseg=512, noverlap=256)

    return out_time


# ---------- Metrics ----------
import torch
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
import numpy as np


class AudioMetrics:
    def __init__(self, fs=16000):
        self.fs = fs
        # Initialize metrics
        # PESQ requires 'wb' (wideband) for 16kHz
        self.pesq = PerceptualEvaluationSpeechQuality(fs, 'wb')
        self.estoi = ShortTimeObjectiveIntelligibility(fs, extended=True)
        self.si_sdr = ScaleInvariantSignalDistortionRatio()

    def compute_all(self, clean, enhanced):
        """
        Computes PESQ, ESTOI, and SI-SDR.
        Args:
            clean (np.array): Reference clean signal (1D)
            enhanced (np.array): Enhanced signal to evaluate (1D)
        Returns:
            dict: Dictionary containing the scores
        """
        # Convert to torch tensors
        clean_t = torch.tensor(clean, dtype=torch.float32)
        est_t = torch.tensor(enhanced, dtype=torch.float32)

        # Ensure shapes are (Batch, Time) -> (1, Time)
        if clean_t.ndim == 1:
            clean_t = clean_t.unsqueeze(0)
            est_t = est_t.unsqueeze(0)

        results = {}

        # Calculate PESQ
        try:
            results['PESQ'] = self.pesq(est_t, clean_t).item()
        except Exception as e:
            print(f"PESQ Error: {e}")
            results['PESQ'] = 0.0

        # Calculate ESTOI
        try:
            results['ESTOI'] = self.estoi(est_t, clean_t).item()
        except Exception as e:
            print(f"ESTOI Error: {e}")
            results['ESTOI'] = 0.0

        # Calculate SI-SDR
        try:
            results['SI_SDR'] = self.si_sdr(est_t, clean_t).item()
        except Exception as e:
            print(f"SI-SDR Error: {e}")
            results['SI_SDR'] = -np.inf

        return results
