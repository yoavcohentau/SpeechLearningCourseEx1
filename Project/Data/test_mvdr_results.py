import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pesq import pesq
from pystoi import stoi
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
import torch
from tqdm import tqdm

# -------- הגדרות נתיבים --------
clean_dir = r"J:\My Drive\Courses\YoavAndItayShared\Speech\DataSet_train\clean"
# שתי תיקיות התוצאות להשוואה
enhanced_folders = {
    'MVDR Method A': r"J:\My Drive\Courses\YoavAndItayShared\Speech\DataSet_train\mvdr_out",
    'MVDR Method B': r"J:\My Drive\Courses\YoavAndItayShared\Speech\DataSet_train\bf_cache\wavs"
}

START_IDX = 0
END_IDX = 200
sisdr_metric = ScaleInvariantSignalDistortionRatio()


def get_metrics_for_folder(folder_path):
    results = {'PESQ': [], 'ESTOI': [], 'SI-SDR': []}

    for i in tqdm(range(START_IDX, END_IDX + 1), desc=f"Processing {os.path.basename(folder_path)}"):
        filename = f"{i:06d}.wav"
        ref_path = os.path.join(clean_dir, filename)
        deg_path = os.path.join(folder_path, filename)

        if not (os.path.exists(ref_path) and os.path.exists(deg_path)):
            continue

        try:
            ref, sr = librosa.load(ref_path, sr=16000)
            deg, _ = librosa.load(deg_path, sr=16000)

            # PESQ
            results['PESQ'].append(pesq(16000, ref, deg, 'wb'))
            # ESTOI
            results['ESTOI'].append(stoi(ref, deg, 16000, extended=True))
            # SI-SDR
            ref_t = torch.tensor(ref).unsqueeze(0)
            deg_t = torch.tensor(deg).unsqueeze(0)
            results['SI-SDR'].append(sisdr_metric(deg_t, ref_t).item())

        except Exception as e:
            pass  # מדלג על קבצים בעייתיים

    # מחזיר ממוצעים לכל מדד בתיקייה הזו
    return {m: np.mean(val) for m, val in results.items()}


# -------- הרצת החישובים --------
all_stats = {}
for name, path in enhanced_folders.items():
    print(f"\nEvaluating {name}...")
    all_stats[name] = get_metrics_for_folder(path)


# -------- יצירת גרף השוואתי --------
def plot_comparison(stats_dict):
    methods = list(stats_dict.keys())
    metrics = ['PESQ', 'ESTOI']#, 'SI-SDR']

    x = np.arange(len(metrics))  # מיקומי המדדים
    width = 0.35  # רוחב העמודות

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, method_name in enumerate(methods):
        means = [stats_dict[method_name][m] for m in metrics]
        offset = (i - (len(methods) - 1) / 2) * width
        rects = ax.bar(x + offset, means, width, label=method_name)
        ax.bar_label(rects, padding=3, fmt='%.2f', fontweight='bold')

    ax.set_ylabel('Scores')
    ax.set_title('Comparison of MVDR Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('comparison_results.png')
    plt.show()


plot_comparison(all_stats)

# הדפסת טבלה מספרית
print("\n" + "=" * 30)
print(f"{'Method':<20} | {'PESQ':<7} | {'ESTOI':<7} | {'SI-SDR':<7}")
print("-" * 50)
for name, stats in all_stats.items():
    print(f"{name:<20} | {stats['PESQ']:<7.3f} | {stats['ESTOI']:<7.3f} | {stats['SI-SDR']:<7.3f}")