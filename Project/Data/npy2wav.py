import numpy as np
import soundfile as sf
from tqdm import tqdm


def npy_to_wav(npy_path, wav_path, sample_rate=16000):
    """
    Load a .npy file and save it as a .wav audio file.

    Parameters:
    -----------
    npy_path : str
        Path to input .npy file (should contain audio array).
    wav_path : str
        Path to output .wav file.
    sample_rate : int
        Sampling rate for the output wav file.
    """

    # Load numpy array
    audio = np.load(npy_path)

    # If stereo/multi-channel stored as [channels, time], transpose to [time, channels]
    if audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
        audio = audio.T

    # Normalize if needed (avoid clipping)
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val

    # Save as WAV
    sf.write(wav_path, audio, sample_rate)


START_IDX = 0
END_IDX = 100

# -------- PROCESS FILES BY RANGE --------
print(f"Starting processing from index {START_IDX} to {END_IDX}...")

for i in tqdm(range(START_IDX, END_IDX + 1)):
    sample = f"{i:06d}"
    npy_path = rf"J:\My Drive\Courses\YoavAndItayShared\Speech\DataSet_train\bf_cache" + rf"\{sample}.npy"
    wav_path = rf"J:\My Drive\Courses\YoavAndItayShared\Speech\DataSet_train\bf_cache\wavs" + rf"\{sample}.wav"
    npy_to_wav(npy_path, wav_path, sample_rate=16000)
