import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

data_set_name = "dev-clean"  # "dev-clean" or "test-clean"

data_set_path = fr"J:\My Drive\Courses\2026A\Signal Processing and Machine Learning for Speech\HW\SpeechLearningCourseEx1\data\{data_set_name}\LibriSpeech"

speaker_id = '84'
chapter_number = '121123'
utterance_number = '0000'
file_ext = 'flac'

file_name = f'{speaker_id}-{chapter_number}-{utterance_number}.{file_ext}'

file_path = rf'{data_set_path}\{data_set_name}\{speaker_id}\{chapter_number}\{file_name}'

# load audio
y, sr = librosa.load(file_path, sr=None)


# --- (A) signal characteristics ---
print(f"Sample rate:    {sr}[Hz]")
print(f"Duration:       {len(y)/sr}[sec]")
print(f"Max amp value:  {max(y)}")
print(f"Min amp value:  {min(y)}")


# --- (B) plot waveform (time domain) ---
plt.figure()

plt.subplot(3, 1, 1)
librosa.display.waveshow(y, sr=sr)
plt.title(f"Speech waveform: {file_name}")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()


# --- (C) STFT with window 50 ms and hop 3 ms ---
win_ms = 50e-3   # 50 milliseconds
hop_ms = 3e-3    # 3 milliseconds

n_fft = int(round(sr * win_ms))
hop_length = int(round(sr * hop_ms))

D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window='hann', center=True)
S_mag = np.abs(D)  # magnitude
S_db = librosa.amplitude_to_db(S_mag, ref=np.max)  # dB relative to max

plt.subplot(3, 1, 2)
librosa.display.specshow(S_db,
                         sr=sr,
                         hop_length=hop_length,
                         x_axis='time',
                         y_axis='hz',
                         cmap='viridis')   # color map choice (matplotlib default acceptable)
cbar = plt.colorbar(format="%+2.0f dB")
cbar.set_label('Magnitude (dB)')
plt.title("Speech Spectrogram - STFT magnitude (dB)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.tight_layout()
# plt.show()


# --- (D) MFCCs ---
# Compute mel-spectrogram first (power), then MFCC
n_mels = 40
# S_power = librosa.feature.melspectrogram(S=np.abs(D)**2, sr=sr, n_mels=n_mels, fmax=sr/2, hop_length=hop_length)
S_power = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

n_mfcc = 13
mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S_power), n_mfcc=n_mfcc)

print("\n=== MFCCs ===")
print(f"MFCC shape: {mfcc.shape}  -> (n_mfcc, n_frames) = ({n_mfcc}, {mfcc.shape[1]})")

# Plot MFCCs as heatmap
plt.subplot(3, 1, 3)
# plt.figure(figsize=(11, 4.5))
librosa.display.specshow(mfcc, x_axis='time', sr=sr, hop_length=hop_length, cmap='viridis')
plt.colorbar(format="%+2.2f")
plt.title(f"MFCC - {n_mfcc} coefficients")
plt.xlabel("Time (s)")
plt.ylabel("MFCC coefficients")
plt.tight_layout()
plt.show()

# Optionally print mean and std of MFCCs across time
mfcc_means = np.mean(mfcc, axis=1)
mfcc_stds = np.std(mfcc, axis=1)
print("\nMFCC mean (per coefficient):")
for i, (m,s) in enumerate(zip(mfcc_means, mfcc_stds), start=1):
    print(f"  MFCC {i:02d}: mean = {m:.4f}, std = {s:.4f}")





# # --- Plot waveform ---
# plt.figure(figsize=(12, 3))
# librosa.display.waveshow(y, sr=sr)
# plt.title(f"Waveform: {file_name}")
# plt.xlabel("time (sec)")
# plt.ylabel("amplitude ()")
# plt.tight_layout()
# plt.show()
#
# # --- Compute STFT ---
# D = librosa.stft(y, n_fft=1024, hop_length=256)
# S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
#
# # --- Plot STFT ---
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(S_db, sr=sr, hop_length=256, x_axis='time', y_axis='hz')
# plt.colorbar(format="%+2.f dB")
# plt.title(f"STFT (librosa) – {file_name}")
# plt.tight_layout()
# plt.show()
