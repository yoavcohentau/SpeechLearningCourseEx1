import os
import random
import math
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import fftconvolve
from tqdm import tqdm

from rir_generator import generate  # pip install rir-generator


# ==========================
# CONFIG
# ==========================
SAMPLE_RATE = 16000

NUM_MICS = 5
MIC_SPACING = 0.05  # meters (5 cm)
MIC_HEIGHT = 1.7

OUTPUT_LEN_SEC = 4.0
NUM_SAMPLES = 20000

CLEAN_DIR = r"J:\My Drive\Courses\YoavAndItayShared\Speech\DataSet_clean"
NOISE_DIR = r"J:\My Drive\Courses\YoavAndItayShared\Speech\DataSet_musan\musan\noise\mix"
OUTPUT_DIR = r"J:\My Drive\Courses\YoavAndItayShared\Speech\DataSet_train"

# Room ranges (meters)
ROOM_L_RANGE = (2.0, 6.0)
ROOM_W_RANGE = (3.0, 7.0)
ROOM_H_RANGE = (2.0, 4.0)

# Reverberation time range (seconds)
T60_RANGE = (0.1, 0.7)

# SNR range (dB)
SNR_RANGE = (-6.0, 6.0)

# Minimum margin from walls (meters)
WALL_MARGIN = 0.8

# DOA separation constraint (degrees)
DOA_DIFF_RANGE = (90.0, 180.0)

# Save extra components (recommended for oracle MVDR training)
SAVE_MULTI_COMPONENTS = True  # clean_mc and noise_mc


# ==========================
# Utilities
# ==========================
def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for sub in ["mixture", "clean", "noise"]:
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)
    if SAVE_MULTI_COMPONENTS:
        os.makedirs(os.path.join(OUTPUT_DIR, "clean_mc"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "noise_mc"), exist_ok=True)


def list_wavs(folder):
    files = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(".wav"):
            files.append(os.path.join(folder, fn))
    if not files:
        raise RuntimeError(f"No .wav files found in: {folder}")
    return files


def list_flac(folder):
    files = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(".flac"):
            files.append(os.path.join(folder, fn))
    if not files:
        raise RuntimeError(f"No .flac files found in: {folder}")
    return files


def match_length(x, target_len):
    if len(x) >= target_len:
        return x[:target_len]
    return np.pad(x, (0, target_len - len(x)))


def rms(x, eps=1e-12):
    return np.sqrt(np.mean(x * x) + eps)


def mix_at_snr(clean, noise, snr_db):
    """
    Returns mixture = clean + noise_scaled such that SNR(clean : noise_scaled) = snr_db
    """
    clean_rms = rms(clean)
    noise_rms = rms(noise)
    if noise_rms < 1e-10:
        return clean.copy()

    desired_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    scale = desired_noise_rms / noise_rms
    return clean + noise * scale


def random_room():
    return [
        random.uniform(*ROOM_L_RANGE),
        random.uniform(*ROOM_W_RANGE),
        random.uniform(*ROOM_H_RANGE),
    ]


def random_position(room):
    # Keep away from walls
    x = random.uniform(WALL_MARGIN, max(WALL_MARGIN, room[0] - WALL_MARGIN))
    y = random.uniform(WALL_MARGIN, max(WALL_MARGIN, room[1] - WALL_MARGIN))
    z = random.uniform(1.0, min(MIC_HEIGHT + 0.5, room[2] - 0.5))
    return [x, y, z]


def create_ula(center, num_mics, spacing):
    """
    ULA along x-axis centered at 'center'. Returns array shape (M,3).
    """
    mic_positions = []
    # Centered indices: e.g., M=6 -> offsets [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]*spacing
    offsets = [(i - (num_mics - 1) / 2.0) * spacing for i in range(num_mics)]
    for off in offsets:
        mic_positions.append([center[0] + off, center[1], center[2]])
    return np.array(mic_positions, dtype=np.float64)


def azimuth_angle_deg(src, center):
    dx = src[0] - center[0]
    dy = src[1] - center[1]
    return math.degrees(math.atan2(dy, dx))


def angular_difference_deg(a1, a2):
    diff = abs(a1 - a2)
    return min(diff, 360.0 - diff)


def sample_noise_position_with_doa(room, mic_center, source_angle_deg):
    lo, hi = DOA_DIFF_RANGE
    for _ in range(2000):  # avoid infinite loop
        p = random_position(room)
        ang = azimuth_angle_deg(p, mic_center)
        d = angular_difference_deg(source_angle_deg, ang)
        if lo <= d <= hi:
            return p
    # If we somehow fail, relax constraint minimally
    return random_position(room)


def safe_generate_rir(fs, room, mic_positions, source_pos, t60):
    """
    Wrapper around rir_generator.generate.
    Returns list/array of RIRs per mic: shape (M, rir_len)
    """
    # rir_generator expects:
    # - r: microphone positions (M,3)
    # - s: source position (3,)
    # - L: room dims (3,)
    # The output is typically (M, N)
    rir = generate(
        c=343.0,
        fs=fs,
        r=mic_positions,
        s=source_pos,
        L=room,
        reverberation_time=t60,
    )
    rir = np.array(rir)
    if rir.ndim == 1:
        rir = rir[None, :]
    if rir.shape[0] != mic_positions.shape[0]:
        # Sometimes libraries return transposed; fix if needed.
        if rir.shape[1] == mic_positions.shape[0]:
            rir = rir.T
        else:
            raise RuntimeError(f"Unexpected RIR shape: {rir.shape}")
    return rir


# ==========================
# Main
# ==========================
def main():
    ensure_dirs()

    clean_files = list_flac(CLEAN_DIR)
    noise_files = list_wavs(NOISE_DIR)

    target_len = int(SAMPLE_RATE * OUTPUT_LEN_SEC)

    for i in tqdm(range(NUM_SAMPLES), desc="Generating"):
        try:
            clean_path = random.choice(clean_files)
            noise_path = random.choice(noise_files)

            clean, _ = librosa.load(clean_path, sr=SAMPLE_RATE, mono=True)
            noise, _ = librosa.load(noise_path, sr=SAMPLE_RATE, mono=True)

            clean = match_length(clean, target_len)
            noise = match_length(noise, target_len)

            # Random room and array
            room = random_room()
            t60 = random.uniform(*T60_RANGE)

            mic_center = [room[0] / 2.0, room[1] / 2.0, MIC_HEIGHT]
            mic_positions = create_ula(mic_center, NUM_MICS, MIC_SPACING)

            # Random source position and constrained noise position
            source_pos = random_position(room)
            source_angle = azimuth_angle_deg(source_pos, mic_center)

            noise_pos = sample_noise_position_with_doa(room, mic_center, source_angle)

            # Generate RIRs
            rir_speech = safe_generate_rir(SAMPLE_RATE, room, mic_positions, source_pos, t60)
            rir_noise = safe_generate_rir(SAMPLE_RATE, room, mic_positions, noise_pos, t60)

            # Convolve to get multi-channel components
            speech_mc = np.zeros((NUM_MICS, target_len), dtype=np.float32)
            noise_mc = np.zeros((NUM_MICS, target_len), dtype=np.float32)

            for m in range(NUM_MICS):
                s_m = fftconvolve(clean, rir_speech[m], mode="full")[:target_len]
                n_m = fftconvolve(noise, rir_noise[m], mode="full")[:target_len]
                speech_mc[m] = s_m.astype(np.float32)
                noise_mc[m] = n_m.astype(np.float32)

            # Mix at random SNR per channel (same snr_db used for all mics)
            snr_db = random.uniform(*SNR_RANGE)
            mixture = np.zeros((NUM_MICS, target_len), dtype=np.float32)
            for m in range(NUM_MICS):
                mixture[m] = mix_at_snr(speech_mc[m], noise_mc[m], snr_db).astype(np.float32)

            # Peak normalize to avoid clipping (optional but recommended)
            peak = np.max(np.abs(mixture))
            if peak > 0.99:
                mixture /= (peak + 1e-12)
                speech_mc /= (peak + 1e-12)
                noise_mc /= (peak + 1e-12)

            # Save WAVs (soundfile expects shape (T, C))
            sf.write(os.path.join(OUTPUT_DIR, "mixture", f"{i:06d}.wav"),
                     mixture.T, SAMPLE_RATE)

            sf.write(os.path.join(OUTPUT_DIR, "clean", f"{i:06d}.wav"),
                     clean.astype(np.float32), SAMPLE_RATE)

            # Save reference-channel reverberant noise (optional convenience)
            sf.write(os.path.join(OUTPUT_DIR, "noise", f"{i:06d}.wav"),
                     noise_mc[0].astype(np.float32), SAMPLE_RATE)

            if SAVE_MULTI_COMPONENTS:
                sf.write(os.path.join(OUTPUT_DIR, "clean_mc", f"{i:06d}.wav"),
                         speech_mc.T, SAMPLE_RATE)
                sf.write(os.path.join(OUTPUT_DIR, "noise_mc", f"{i:06d}.wav"),
                         noise_mc.T, SAMPLE_RATE)
        except Exception as e:
            print(e)

    print(f"Done. Wrote {NUM_SAMPLES} samples to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
















# import os
# import random
# import numpy as np
# import soundfile as sf
# import librosa
# from scipy.signal import fftconvolve
# from rir_generator import generate
# from tqdm import tqdm
#
# # ==========================
# # CONFIG
# # ==========================
#
# SAMPLE_RATE = 16000
# NUM_MICS = 5
# MIC_SPACING = 0.05  # 5 cm
# OUTPUT_LEN_SEC = 4
# OUTPUT_DIR = r"J:\My Drive\Courses\YoavAndItayShared\Speech\DataSet_train"
# NUM_SAMPLES = 5000  # how many mixtures to generate
#
# CLEAN_DIR = r"J:\My Drive\Courses\YoavAndItayShared\Speech\DataSet_clean"
# NOISE_DIR = r"J:\My Drive\Courses\YoavAndItayShared\Speech\DataSet_musan\musan\noise"
#
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(os.path.join(OUTPUT_DIR, "mixture"), exist_ok=True)
# os.makedirs(os.path.join(OUTPUT_DIR, "clean"), exist_ok=True)
# os.makedirs(os.path.join(OUTPUT_DIR, "noise"), exist_ok=True)
#
# # ==========================
# # Utility Functions
# # ==========================
#
# def random_room():
#     return [
#         random.uniform(5, 10),  # length
#         random.uniform(5, 10),  # width
#         random.uniform(3, 4)    # height
#     ]
#
# def random_position(room):
#     return [
#         random.uniform(1, room[0]-1),
#         random.uniform(1, room[1]-1),
#         random.uniform(1, room[2]-1)
#     ]
#
# def create_ula(center, num_mics, spacing):
#     mic_positions = []
#     for i in range(num_mics):
#         mic_positions.append([
#             center[0] + (i - num_mics//2) * spacing,
#             center[1],
#             center[2]
#         ])
#     return np.array(mic_positions)
#
# def match_length(x, length):
#     if len(x) > length:
#         return x[:length]
#     else:
#         return np.pad(x, (0, length - len(x)))
#
# def add_snr(clean, noise, snr_db):
#     clean_power = np.mean(clean**2)
#     noise_power = np.mean(noise**2)
#     scale = np.sqrt(clean_power / (10**(snr_db/10) / noise_power))
#     return clean + noise * scale
#
# # ==========================
# # Load file lists
# # ==========================
#
# clean_files = [os.path.join(CLEAN_DIR, f) for f in os.listdir(CLEAN_DIR)]
# noise_files = [os.path.join(NOISE_DIR, f) for f in os.listdir(NOISE_DIR)]
#
# # ==========================
# # Main Loop
# # ==========================
#
# for i in tqdm(range(NUM_SAMPLES)):
#
#     # --- Random selection
#     clean_path = random.choice(clean_files)
#     noise_path = random.choice(noise_files)
#
#     clean, _ = librosa.load(clean_path, sr=SAMPLE_RATE)
#     noise, _ = librosa.load(noise_path, sr=SAMPLE_RATE)
#
#     target_len = SAMPLE_RATE * OUTPUT_LEN_SEC
#     clean = match_length(clean, target_len)
#     noise = match_length(noise, target_len)
#
#     # --- Random room config
#     room = random_room()
#     t60 = random.uniform(0.1, 0.7)
#
#     mic_center = [room[0]/2, room[1]/2, 1.5]
#     mic_positions = create_ula(mic_center, NUM_MICS, MIC_SPACING)
#
#     source_pos = random_position(room)
#     noise_pos = random_position(room)
#
#     # --- Generate RIRs
#     rir_speech = generate(
#         c=343,
#         fs=SAMPLE_RATE,
#         r=mic_positions,
#         s=source_pos,
#         L=room,
#         reverberation_time=t60
#     )
#
#     rir_noise = generate(
#         c=343,
#         fs=SAMPLE_RATE,
#         r=mic_positions,
#         s=noise_pos,
#         L=room,
#         reverberation_time=t60
#     )
#
#     # --- Convolve
#     multi_speech = []
#     multi_noise = []
#
#     for m in range(NUM_MICS):
#         speech_m = fftconvolve(clean, rir_speech[m])[:target_len]
#         noise_m = fftconvolve(noise, rir_noise[m])[:target_len]
#         multi_speech.append(speech_m)
#         multi_noise.append(noise_m)
#
#     multi_speech = np.stack(multi_speech)
#     multi_noise = np.stack(multi_noise)
#
#     # --- Random SNR
#     snr_db = random.uniform(-6, 6)
#     mixture = []
#
#     for m in range(NUM_MICS):
#         mixture_m = add_snr(multi_speech[m], multi_noise[m], snr_db)
#         mixture.append(mixture_m)
#
#     mixture = np.stack(mixture)
#
#     # --- Save
#     sf.write(os.path.join(OUTPUT_DIR, "mixture", f"{i}.wav"),
#              mixture.T, SAMPLE_RATE)
#
#     sf.write(os.path.join(OUTPUT_DIR, "clean", f"{i}.wav"),
#              clean, SAMPLE_RATE)
#
#     sf.write(os.path.join(OUTPUT_DIR, "noise", f"{i}.wav"),
#              multi_noise[0], SAMPLE_RATE)
#
# print("Dataset generation complete.")