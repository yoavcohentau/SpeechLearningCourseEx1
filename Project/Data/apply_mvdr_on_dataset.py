import os
import numpy as np
import soundfile as sf
from tqdm import tqdm

from Ex2.Q2_func import apply_mvdr

# -------- CONFIGURATION --------
mixture_dir = r"J:\My Drive\Courses\YoavAndItayShared\Speech\DataSet_valid\mixture"
noise_dir = r"J:\My Drive\Courses\YoavAndItayShared\Speech\DataSet_valid\noise_mc"
output_dir = r"J:\My Drive\Courses\YoavAndItayShared\Speech\DataSet_valid\mvdr_out"

# הגדרת הטווח (למשל מ-1 עד 300)
START_IDX = 30000
END_IDX = 30879

os.makedirs(output_dir, exist_ok=True)

# -------- PROCESS FILES BY RANGE --------
print(f"Starting processing from index {START_IDX} to {END_IDX}...")

for i in tqdm(range(START_IDX, END_IDX + 1)):
    # יצירת שם הקובץ בפורמט של 6 ספרות (000001.wav)
    filename = f"{i:06d}.wav"

    mix_path = os.path.join(mixture_dir, filename)
    noise_path = os.path.join(noise_dir, filename)

    # בדיקה אם קובץ ה-mixture קיים (אם לא, פשוט מדלגים למספר הבא)
    if not os.path.exists(mix_path):
        continue

    # בדיקה אם קובץ ה-noise קיים
    if not os.path.exists(noise_path):
        print(f"\n[Warning] Noise file missing for {filename}, skipping.")
        continue

    try:
        # Load (shape: samples x channels)
        mix, sr = sf.read(mix_path)
        noise, sr2 = sf.read(noise_path)

        if sr != sr2:
            print(f"\n[Error] Sample rates mismatch for {filename}.")
            continue

        # Convert to shape (M, T) - כפי ש-apply_mvdr מצפה
        mix = mix.T
        noise = noise.T

        # Apply MVDR
        enhanced = apply_mvdr(mix, noise)

        # Normalize to prevent clipping (0.9 כדי להשאיר Headroom)
        max_val = np.max(np.abs(enhanced))
        if max_val > 1.0:
            enhanced = enhanced / (max_val + 1e-8)

        # Save output
        out_path = os.path.join(output_dir, filename)
        sf.write(out_path, enhanced, sr)

    except Exception as e:
        print(f"\n[Error] Failed to process {filename}: {str(e)}")

print("\nDone processing all available files in range.")








# import os
# import glob
# import numpy as np
# import soundfile as sf
#
# from Ex2.Q2_func import apply_mvdr
#
# # -------- PATHS --------
# mixture_dir = r"J:\My Drive\Courses\YoavAndItayShared\Speech\DataSet_train\mixture"
# noise_dir = r"J:\My Drive\Courses\YoavAndItayShared\Speech\DataSet_train\noise_mc"
# output_dir = r"J:\My Drive\Courses\YoavAndItayShared\Speech\DataSet_train\mvdr_out"
#
# os.makedirs(output_dir, exist_ok=True)
#
# # -------- PROCESS FILES --------
# mixture_files = glob.glob(os.path.join(mixture_dir, "*.wav"))
#
# for mix_path in mixture_files:
#
#     filename = os.path.basename(mix_path)
#     noise_path = os.path.join(noise_dir, filename)
#
#     if not os.path.exists(noise_path):
#         print(f"Noise file missing for {filename}, skipping.")
#         continue
#
#     # Load mixture (shape: samples x channels)
#     mix, sr = sf.read(mix_path)
#     noise, sr2 = sf.read(noise_path)
#
#     if sr != sr2:
#         raise ValueError("Sample rates do not match.")
#
#     # Convert to shape (M, T)
#     mix = mix.T
#     noise = noise.T
#
#     # Apply MVDR
#     enhanced = apply_mvdr(mix, noise)
#
#     # Normalize to prevent clipping
#     max_val = np.max(np.abs(enhanced))
#     if max_val > 1:
#         enhanced = enhanced / max_val
#
#     # Save output
#     out_path = os.path.join(output_dir, filename)
#     sf.write(out_path, enhanced, sr)
#
#     print(f"Saved: {out_path}")
#
# print("Done.")
