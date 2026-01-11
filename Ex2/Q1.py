import numpy as np
import matplotlib.pyplot as plt
from rir_generator import generate
import librosa
from scipy.signal import fftconvolve


PLOT_FLAG = True

DATA_SET_NAME = "dev-clean"  # "dev-clean" or "test-clean"
DATA_SET_PATH = fr"J:\My Drive\Courses\2026A\Signal Processing and Machine Learning for Speech\HW\HW1\SpeechLearningCourseEx1\data\{DATA_SET_NAME}\LibriSpeech"


def generate_room_impulse_responses(
    fs,
    room_dim,
    mic_center,
    num_mics,
    mic_spacing,
    source_angle_deg,
    source_distance,
    T60_values
):
    # Convert inputs to numpy arrays
    mic_center = np.array(mic_center)

    # Compute microphone positions (linear array along x-axis)
    mic_positions = [
        mic_center + np.array([(i - (num_mics - 1) / 2) * mic_spacing, 0, 0])
        for i in range(num_mics)
    ]

    # Compute source position
    theta = np.deg2rad(source_angle_deg)
    source_pos = mic_center + source_distance * np.array([
        np.cos(theta),
        np.sin(theta),
        0
    ])

    rirs = {}
    for T60 in T60_values:
        n_samples = int(T60 * fs)

        # Generate RIRs for all microphones
        rir_list = generate(
            c=343,
            fs=fs,
            r=mic_positions,
            s=source_pos,
            L=room_dim,
            reverberation_time=T60,
            nsample=n_samples
        )
        rirs[T60] = rir_list.T

    return rirs


def generate_microphone_signals(
    clean_speech_path,
    fs,
    rirs
):
    # Load clean speech using librosa
    speech, fs_file = librosa.load(
        clean_speech_path,
        sr=fs,
        mono=True
    )

    mic_signals = {}
    for T60, rir_list in rirs.items():
        num_mics = len(rir_list)
        signal_length = len(speech)

        # Allocate array for microphone signals
        X = np.zeros((num_mics, signal_length), dtype=np.float32)

        for m in range(num_mics):
            # Convolve clean speech with RIR of microphone m
            x_m = fftconvolve(speech, rir_list[m])

            # Truncate to original signal length
            X[m, :] = x_m[:signal_length]

        mic_signals[T60] = X

    return mic_signals


def main_q1():
    # General parameters
    fs = 16000
    room_dim = [4, 5, 3]
    T60_values = [0.15, 0.30]

    # Microphone array parameters
    mic_center = [2, 1, 1.7]
    num_mics = 5
    mic_spacing = 0.05

    # Source parameters
    source_angle_deg = 30
    source_distance = 1.5

    # Section (a): Generate RIRs
    rirs = generate_room_impulse_responses(
        fs=fs,
        room_dim=room_dim,
        mic_center=mic_center,
        num_mics=num_mics,
        mic_spacing=mic_spacing,
        source_angle_deg=source_angle_deg,
        source_distance=source_distance,
        T60_values=T60_values
    )

    # Plot time-domain RIR for the first microphone
    if PLOT_FLAG:
        for T60_val in T60_values:
            rir_mic1 = rirs[T60_val][0]
            rir_mic1 = rir_mic1 / np.max(np.abs(rir_mic1))  # Normalize for visualization
            t = np.arange(len(rir_mic1)) / fs

            plt.plot(t, rir_mic1, label=f"T60 = {int(T60_val*1000)} ms")

        plt.xlabel("Time [s]")
        plt.ylabel("Normalized amplitude")
        plt.title("Room Impulse Responses – First Microphone")
        plt.legend()
        plt.grid()
        plt.show()

    # Section (b)
    speaker_id = '84'
    chapter_number = '121123'
    utterance_number = '0000'
    file_ext = 'flac'

    file_name = f'{speaker_id}-{chapter_number}-{utterance_number}.{file_ext}'

    file_path = rf'{DATA_SET_PATH}\{DATA_SET_NAME}\{speaker_id}\{chapter_number}\{file_name}'

    mic_signals = generate_microphone_signals(
        clean_speech_path=file_path,
        fs=fs,
        rirs=rirs
    )

    # Sanity check (outside functions)
    for T60 in mic_signals:
        print(
            f"T60 = {int(T60*1000)} ms, "
            f"mic_signals shape = {mic_signals[T60].shape}"
        )

    T60_test = 0.30
    mic_index = 0

    plt.plot(mic_signals[T60_test][mic_index])
    plt.title("Reverberant speech – Microphone 1")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

    # Further sections will be added here later


if __name__ == "__main__":
    main_q1()


# import numpy as np
# import matplotlib.pyplot as plt
# from rir_generator import generate
#
# # Sampling rate
# fs = 16000
#
# # Room dimensions [x, y, z] in meters
# room_dim = [4, 5, 3]
#
# # Microphone array center
# mic_center = np.array([2, 1, 1.7])
#
# # Distance between adjacent microphones (5 cm)
# d = 0.05
#
# # First microphone position (leftmost)
# mic_pos = mic_center + np.array([-2*d, 0, 0])
#
# # Source position: 30 degrees, 1.5 m from array center
# theta = np.deg2rad(30)
# source_pos = mic_center + 1.5 * np.array([
#     np.cos(theta),
#     np.sin(theta),
#     0
# ])
#
# # Reverberation times to evaluate
# T60_values = [0.15, 0.30]
#
# plt.figure(figsize=(10, 4))
#
# for T60 in T60_values:
#     # Number of samples for the RIR
#     n_samples = int(T60 * fs)
#
#     # Generate room impulse response
#     rir = generate(
#         c=343,                 # Speed of sound [m/s]
#         fs=fs,                 # Sampling rate
#         r=[mic_pos],           # First microphone position
#         s=source_pos,          # Source position
#         L=room_dim,            # Room dimensions
#         reverberation_time=T60,
#         nsample=n_samples
#     )
#
#     # Normalize RIR for visualization
#     rir = rir / np.max(np.abs(rir))
#
#     # Time axis
#     t = np.arange(len(rir)) / fs
#
#     # Plot RIR in time domain
#     plt.plot(t, rir, label=f"T60 = {int(T60*1000)} ms")
#
# plt.xlabel("Time [s]")
# plt.ylabel("Normalized amplitude")
# plt.title("Room Impulse Response – First Microphone")
# plt.legend()
# plt.grid()
# plt.show()
