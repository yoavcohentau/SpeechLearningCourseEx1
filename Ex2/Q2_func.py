import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.linalg import eigh
from scipy.linalg import pinv
from scipy.signal import stft, istft
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio import ShortTimeObjectiveIntelligibility


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
def estimate_cov_matrix(stft_mat):
    X = np.asarray(stft_mat, dtype=np.complex128)
    t = X.shape[2]
    R = np.einsum('mft,nft->fmn', X, X.conj()) / t

    return R


def estimate_rtf_using_gevd(noisy_stft, noise_cov, ref_mic=2):
    R_y = estimate_cov_matrix(noisy_stft)

    # find the rtf per frequency
    n_freqs, n_channels = R_y.shape[0], R_y.shape[1]
    rtf = np.zeros((n_channels, n_freqs), dtype=np.complex128)
    for f in range(n_freqs):
        R_n = noise_cov[f] + 1e-3 * np.eye(n_channels)
        _, vecs = eigh(R_y[f], R_n)  # calc GEVD
        v = vecs[:, -1]
        h = R_n @ v  # calc RTF
        rtf[:, f] = h / (h[ref_mic] + 1e-12)  # normalize by ref mic

    return rtf


def _compute_mvdr_weights(noise_cov, rtf):
    n_freqs, n_mics, _ = noise_cov.shape
    weights = np.zeros((n_mics, n_freqs), dtype=np.complex128)

    for f in range(n_freqs):
        R_inv = pinv(noise_cov[f])
        h = rtf[:, f]

        numerator = R_inv @ h  # MVDR Numerator: R^-1 * h
        denominator = (h.conj().T @ numerator).real  # MVDR Denominator: h^H * R^-1 * h

        if denominator < 1e-15:
            weights[:, f] = 0
        else:
            weights[:, f] = numerator / denominator

    return weights


def apply_mvdr(mic_signals, noise, win_length=800, hop_length=48):
    mic_signals_stft = librosa.stft(mic_signals, n_fft=win_length, hop_length=hop_length, win_length=win_length)
    noise_stft = librosa.stft(noise, n_fft=win_length, hop_length=hop_length, win_length=win_length)

    noise_cov = estimate_cov_matrix(noise_stft)

    rtf = estimate_rtf_using_gevd(mic_signals_stft, noise_cov)
    # w = mvdr_weights(noise_cov, rtf)
    w = _compute_mvdr_weights(noise_cov, rtf)

    X = np.asarray(mic_signals_stft, dtype=np.complex128)
    w = np.asarray(w, dtype=np.complex128)
    M, F, N = X.shape
    out_stft = np.zeros((F, N), dtype=np.complex128)
    for f in range(F):  # apply filter
        out_stft[f, :] = w[:, f].conj().T @ X[:, f, :]

    out_mvdr = librosa.istft(
        out_stft, hop_length=hop_length, win_length=win_length, n_fft=win_length, length=np.shape(mic_signals)[1])

    return np.real(out_mvdr)


# ---------- Metrics ----------
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


def parse_and_plot_results(all_metrics):
    """
    Parses a LIST of dictionaries.
    Each dictionary contains keys formatted as: 'Algo-NoiseType-SNR-T60-ExampleIdx'
    """
    # --- שלב 1: חילוץ הנתונים לטבלה שטוחה ---
    parsed_data = []

    # רצים על הרשימה הראשית (כל איבר הוא מילון של דוגמה אחת)
    for example_dict in all_metrics:
        # רצים על המפתחות בתוך המילון של הדוגמה הנוכחית
        for key, scores in example_dict.items():
            # פירוק המפתח: f'{Algo}-{Noise}-{snr}-{T60}-{example_idx}'
            # דוגמה מהתמונה: 'DSB-white-0-0.15-0'
            parts = key.split('-')

            # הגנה בסיסית
            if len(parts) < 5:
                continue

            algo = parts[0]  # DSB
            noise_type = parts[1]  # white
            snr = float(parts[2])  # 0 (ממירים למספר למיון)
            t60 = float(parts[3])  # 0.15 (ממירים למספר למיון)

            # יצירת שורות לטבלה (שורה לכל מדד כדי שיהיה נוח לצייר)
            # PESQ
            parsed_data.append({
                'Algorithm': algo,
                'NoiseType': noise_type,
                'SNR': snr,
                'T60': t60,
                'Condition': f"T60={t60}, SNR={int(snr)}dB",  # עמודת עזר לציר X
                'Metric': 'PESQ',
                'Score': scores.get('PESQ', 0)
            })
            # ESTOI
            parsed_data.append({
                'Algorithm': algo,
                'NoiseType': noise_type,
                'SNR': snr,
                'T60': t60,
                'Condition': f"T60={t60}, SNR={int(snr)}dB",
                'Metric': 'ESTOI',
                'Score': scores.get('ESTOI', 0)
            })
            # SI_SDR
            parsed_data.append({
                'Algorithm': algo,
                'NoiseType': noise_type,
                'SNR': snr,
                'T60': t60,
                'Condition': f"T60={t60}, SNR={int(snr)}dB",
                'Metric': 'SI_SDR',
                'Score': scores.get('SI_SDR', -np.inf)
            })

    # המרה ל-DataFrame של Pandas
    df = pd.DataFrame(parsed_data)

    # מיון לפי T60 ו-SNR כדי שציר ה-X יהיה מסודר הגיונית
    df.sort_values(by=['T60', 'SNR'], inplace=True)

    df.to_csv('metrics.csv', index=False)

    # --- שלב 2: ציור הגרפים ---

    # הגדרת סגנון
    sns.set_theme(style="whitegrid")

    # יצירת גריד של גרפים:
    # שורות (row) = מדדים (PESQ, ESTOI, SI_SDR)
    # עמודות (col) = סוג רעש (white, inter)
    g = sns.catplot(
        data=df,
        kind="bar",
        x="Condition",  # ציר ה-X: התנאים (T60, SNR)
        y="Score",  # ציר ה-Y: הציון (הממוצע מחושב אוטומטית ע"י seaborn)
        hue="Algorithm",  # צבעים: האלגוריתמים (DSB, MVDR, Denoiser)
        col="NoiseType",  # עמודה לכל סוג רעש
        row="Metric",  # שורה לכל מדד
        height=3.5,
        aspect=1.6,
        sharey=False,  # ציר Y נפרד לכל שורה (כי הסקאלות שונות)
        palette="viridis",  # צבעים יפים
        errorbar='sd',  # מציג את סטיית התקן כקו שחור (אופציונלי)
        capsize=0.1
    )

    # --- חלק התיקון ---

    # כותרות ועיצוב
    # top=0.9: משאיר מקום לכותרת הראשית
    # bottom=0.2: משאיר מקום למטה לכיתובים המסובבים (זה התיקון החשוב!)
    # hspace=0.3: מוסיף קצת מרווח אנכי בין השורות של הגרפים כדי שהכיתובים לא יתנגשו
    g.fig.subplots_adjust(top=0.9, bottom=0.2, hspace=0.4)

    g.fig.suptitle('Metrics: Averaged over all Examples', fontsize=16)

    g.set_axis_labels("", "Score")
    g.set_titles("{row_name} | {col_name} Noise")

    # סיבוב התוויות למטה כדי שיהיה קריא
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(25)
            label.set_ha('right')

    plt.show()

    # # כותרות ועיצוב
    # g.fig.subplots_adjust(top=0.9)
    # g.fig.suptitle('Performance Comparison: Averaged over all Examples', fontsize=16)
    #
    # g.set_axis_labels("", "Score")
    # g.set_titles("{row_name} | {col_name} Noise")
    #
    # # סיבוב התוויות למטה כדי שיהיה קריא
    # for ax in g.axes.flat:
    #     for label in ax.get_xticklabels():
    #         label.set_rotation(25)
    #         label.set_ha('right')
    #
    # plt.show()


# def plot_aggregate_results(df_results):
#     """
#     Plots bar charts using Matplotlib and Pandas.
#     """
#     metrics = ['PESQ', 'ESTOI', 'SI_SDR']
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#
#     for i, metric in enumerate(metrics):
#         # Filter and Calculate Stats
#         df_metric = df_results[df_results['Metric'] == metric]
#         means = df_metric.groupby(['NoiseType', 'Algorithm'])['Score'].mean().unstack()
#         stds = df_metric.groupby(['NoiseType', 'Algorithm'])['Score'].std().unstack()
#
#         # Plot
#         means.plot(kind='bar', yerr=stds, ax=axes[i], capsize=4, rot=0, alpha=0.8)
#
#         axes[i].set_title(f"{metric}")
#         axes[i].set_ylabel("Score")
#         axes[i].grid(axis='y', linestyle='--', alpha=0.5)
#
#     plt.suptitle("Beamforming Performance Comparison (Mean ± SD)", fontsize=16)
#     plt.tight_layout()
#     plt.show()


