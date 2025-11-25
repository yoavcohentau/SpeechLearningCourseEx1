import os
import numpy as np
import librosa
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from tqdm import tqdm


# Read speakers gender from SPEAKERS.TXT file
def load_speaker_genders(speakers_txt_path):
    genders = {}
    with open(speakers_txt_path, "r") as f:
        for line in f:
            if line.startswith(";"):
                continue
            parts = line.strip().split("|")
            if len(parts) < 3:
                continue
            speaker_id = parts[0].strip()
            gender = parts[1].strip()
            genders[speaker_id] = gender
    return genders


# Read the audio examples for each speaker
def collect_audio_files(base_paths, limit_per_speaker=20):
    """
    base_paths - data sets paths list: [dev-clean, test-clean]
    """
    speaker_to_files = {}

    for base in base_paths:
        for speaker in os.listdir(base):
            spk_path = os.path.join(base, speaker)
            if not os.path.isdir(spk_path):
                continue

            all_files = []
            for chap in os.listdir(spk_path):
                chap_path = os.path.join(spk_path, chap)
                if not os.path.isdir(chap_path):
                    continue
                for file_path in os.listdir(chap_path):
                    if file_path.endswith(".flac"):
                        all_files.append(os.path.join(chap_path, file_path))

            # take 'limit_per_speaker' files in maximum
            all_files = sorted(all_files)[:limit_per_speaker]
            speaker_to_files[speaker] = all_files

    return speaker_to_files


# Create MFCC features
def extract_mfcc_features(audio, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)  # coefficient mean


# Extract pitch features
def extract_pitch_features(audio, sr):
    f0 = librosa.yin(audio, fmin=50, fmax=350, sr=sr, frame_length=1024, hop_length=320)
    f0 = f0[np.isfinite(f0)]  # remove NaN

    if len(f0) == 0:  # if no pitch detected
        return np.zeros(5)

    return np.array([
        np.min(f0),
        np.max(f0),
        np.mean(f0),
        np.median(f0),
        np.var(f0)
    ])


# Create features
def build_feature_dataset(speaker_to_files, genders, feature_type):
    x = []
    y = []

    for speaker, files in tqdm(speaker_to_files.items()):
        if speaker not in genders:
            continue

        gender = genders[speaker]

        for fpath in files:
            audio, sr = librosa.load(fpath, sr=None)

            if feature_type == "mfcc":
                feats = extract_mfcc_features(audio, sr)
            elif feature_type == "pitch":
                feats = extract_pitch_features(audio, sr)
            else:
                raise ValueError("Unknown feature type")

            x.append(feats)
            y.append(gender)

    return np.array(x), np.array(y)


# Split to train and test sets
def split_speakers(speakers, train_size=66):
    speakers = sorted(speakers)
    train_speakers = speakers[:train_size]
    test_speakers = speakers[train_size:]
    return train_speakers, test_speakers


def main():
    # data sets paths
    data_sets_path = fr"J:\My Drive\Courses\2026A\Signal Processing and Machine Learning for Speech\HW\SpeechLearningCourseEx1\data"
    dev_data_path = fr"{data_sets_path}\dev-clean\LibriSpeech\dev-clean"
    test_data_path = fr"{data_sets_path}\test-clean\LibriSpeech\test-clean"
    speakers_txt_path = fr"{data_sets_path}\dev-clean\LibriSpeech\SPEAKERS.TXT"

    # load genders labels
    genders = load_speaker_genders(speakers_txt_path)

    # collect data files paths
    speaker_to_files = collect_audio_files([dev_data_path, test_data_path], limit_per_speaker=20)

    all_speakers = sorted(speaker_to_files.keys())
    train_spk, test_spk = split_speakers(all_speakers)

    train_dict = {spk: speaker_to_files[spk] for spk in train_spk}
    test_dict = {spk: speaker_to_files[spk] for spk in test_spk}

    results = {}

    for feature_type in ["mfcc", "pitch"]:
        print(f"\n--- Extracting features: {feature_type} ---")

        x_train, y_train = build_feature_dataset(train_dict, genders, feature_type)
        x_test, y_test = build_feature_dataset(test_dict, genders, feature_type)

        # SVM model
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))
        ])

        print("Training SVM...")
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)
        acc = accuracy_score(y_test, y_pred)

        results[feature_type] = acc
        print(f"Accuracy ({feature_type}): {acc:.4f}")

    print("\n--- Summary ---")
    print(pd.DataFrame.from_dict(results, orient='index', columns=["Accuracy"]))


if __name__ == "__main__":
    main()
