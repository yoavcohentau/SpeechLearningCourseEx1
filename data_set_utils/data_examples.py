import os
import random
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import stft
import numpy as np

import matplotlib
matplotlib.use('TkAgg')


def load_speakers_metadata(speakers_file):
    speakers = {}
    with open(speakers_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(";") or len(line.strip()) == 0:
                continue
            # format:
            # ID | SEX | SUBSET | MINUTES | NAME
            parts = line.strip().split("|")
            if len(parts) != 5:
                continue
            spk_id = parts[0].strip()
            speakers[spk_id] = {
                "sex": parts[1].strip(),
                "subset": parts[2].strip(),
                "minutes": float(parts[3].strip()),
                "name": parts[4].strip()
            }
    return speakers


def collect_audio_files(base_path):
    """
    Returns a list of tuples:
    (speaker_id, chapter_id, utt_id, filepath)
    """
    base = Path(base_path)
    all_files = []
    for speaker in base.iterdir():
        if not speaker.is_dir():
            continue
        speaker_id = speaker.name

        for chapter in speaker.iterdir():
            if not chapter.is_dir():
                continue
            chapter_id = chapter.name

            for flac in chapter.glob("*.flac"):
                utt_id = flac.stem.split("-")[-1]
                all_files.append((speaker_id, chapter_id, utt_id, str(flac)))
    return all_files


def print_examples(dataset_name, files, speakers_meta, num_examples=2):
    print(f"\n=== Examples from {dataset_name} ===\n")
    examples = random.sample(files, min(num_examples, len(files)))

    for spk, chap, utt, fpath in examples:
        info = speakers_meta.get(spk, {})
        print(f"Speaker ID: {spk}")
        print(f"  Name: {info.get('name', 'Unknown')}")
        print(f"  Sex: {info.get('sex', 'Unknown')}")
        print(f"  Subset Listed: {info.get('subset', 'Unknown')}")
        print(f"  Total Minutes: {info.get('minutes', 'Unknown')}")
        print(f"Chapter: {chap}, Utterance: {utt}")
        print(f"Audio path: {fpath}")

        # load audio
        waveform, sr = sf.read(fpath)
        print(f"  Audio shape: {waveform.shape}, Sample Rate: {sr}")

        # --- STFT plot ---
        f, t, Zxx = stft(waveform, fs=sr, nperseg=512)

        plt.figure(figsize=(10, 4))
        plt.pcolormesh(t, f, 20 * np.log10(np.abs(Zxx) + 1e-8), shading='gouraud')
        plt.title(f"STFT: Speaker {spk}, Chapter {chap}, Utterance {utt}")
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [sec]")
        plt.colorbar(label="Magnitude (dB)")
        plt.tight_layout()
        plt.show()

        print("-" * 50)


if __name__ == "__main__":
    data_set_name = "test-clean"  # "dev-clean" or "test-clean"

    # EDIT HERE: put your local LibriSpeech path (directory containing dev-clean/, test-clean/, SPEAKERS.TXT)
    LIBRISPEECH_ROOT = fr"J:\My Drive\Courses\2026A\Signal Processing and Machine Learning for Speech\HW\SpeechLearningCourseEx1\data\{data_set_name}\LibriSpeech"

    speakers_file = os.path.join(LIBRISPEECH_ROOT, "SPEAKERS.TXT")
    speakers_meta = load_speakers_metadata(speakers_file)

    data_set_path = os.path.join(LIBRISPEECH_ROOT, data_set_name)

    data_set_files = collect_audio_files(data_set_path)

    print_examples(data_set_name, data_set_files, speakers_meta, num_examples=1)
