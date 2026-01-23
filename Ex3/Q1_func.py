import matplotlib.pyplot as plt
import numpy as np
import rir_generator as rir
from scipy import signal
from scipy.signal import fftconvolve


# ---Room Simulation---
def generate_room_impulse_responses(fs, room_dim, mic_locations, source_location, T60_values):
    c = 344.0  # speed of sound

    # Handle source location: assign random Z (1.2m-2.0m) if only X,Y provided
    if source_location.shape[0] == 2:
        z_coord = np.random.uniform(1.2, 2.0)
        source_pos = np.append(source_location, z_coord)
    else:
        source_pos = source_location

    all_rirs = {}

    for t60 in T60_values:
        # Define RIR length based on T60 and sampling frequency
        nsample = int(t60 * fs)

        # Generate RIRs using Image Source Method (ISM)
        # Resulting 'h' matrix columns correspond to each microphone
        h = rir.generate(
            c=c,
            fs=fs,
            r=np.array(mic_locations),  # Mic array coordinates {r_m} [cite: 25]
            s=source_pos,  # Source position 'p' [cite: 26]
            L=np.array(room_dim),  # Dimensions of the acoustic environment
            reverberation_time=t60,  # Target T60 for simulation
            nsample=nsample  # Total number of RIR samples
        )

        # Store transposed matrix: each row represents a microphone channel [cite: 27]
        all_rirs[t60] = h.T

    return all_rirs


def generate_microphone_signals(
        sound,
        rirs
):
    mic_signals = {}
    for T60, rir_list in rirs.items():
        num_mics = len(rir_list)
        signal_length = len(sound)
        X = np.zeros((num_mics, signal_length), dtype=np.float32)
        for m in range(num_mics):
            # Convolve clean speech with the m RIR
            x_fft = fftconvolve(sound, rir_list[m])
            X[m, :] = x_fft[:signal_length]

        mic_signals[T60] = X

    return mic_signals


# ---utils---
def apply_stft(signals, fs, nperseg=512, noverlap=256):
    stft_data = []
    for i in range(signals.shape[0]):
        f, t, Z = signal.stft(signals[i], fs=fs, window='hamming', nperseg=nperseg, noverlap=noverlap)
        stft_data.append(Z)
    return f, t, np.array(stft_data)


# ---SPR-PHAT---
def compute_gcc_phat(stft_data, n_fft=512):
    num_mics = stft_data.shape[0]
    gcc_channels = {}

    for m1 in range(num_mics):
        for m2 in range(m1 + 1, num_mics):
            cps = stft_data[m1] * np.conj(stft_data[m2])
            phi_phat = cps / (np.abs(cps) + 1e-10)

            avg_phi = np.mean(phi_phat, axis=1)

            r_12 = np.fft.irfft(avg_phi, n=n_fft)
            r_12 = np.fft.fftshift(r_12)

            gcc_channels[(m1, m2)] = r_12

    return gcc_channels


def compute_srp_map(gcc_channels, fs, room_dim, mic_locations, num_px_x=64, num_px_y=64):
    """
    Builds the SRP-PHAT likelihood map using a fixed number of pixels per dimension.

    Args:
        gcc_channels (dict): GCC-PHAT functions for each pair.
        fs (int): Sampling frequency.
        room_dim (list): Room dimensions [L, W, H].
        mic_locations (np.ndarray): Array of microphone positions.
        num_px_x (int): Number of pixels along the X-axis (length).
        num_px_y (int): Number of pixels along the Y-axis (width).
    """
    c = 344.0  # Speed of sound [cite: 38]
    n_fft = len(next(iter(gcc_channels.values())))  # Get FFT length

    # 1. Create a regular grid (p) using the specified number of pixels [cite: 107]
    # np.linspace ensures we have exactly num_px across the room dimension
    x_range = np.linspace(0, room_dim[0], num_px_x)
    y_range = np.linspace(0, room_dim[1], num_px_y)
    z_fixed = 1.5  # Target height for the 2D scan

    srp_map = np.zeros((num_px_x, num_px_y))

    # 2. Iterate through each pixel coordinate (candidate position p)
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            p = np.array([x, y, z_fixed])
            total_power = 0

            # 3. Sum the GCC-PHAT values for all microphone pairs [cite: 109]
            for (m1, m2), r_12 in gcc_channels.items():
                # Calculate theoretical TDOA for candidate point p
                dist1 = np.linalg.norm(p - mic_locations[m1])
                dist2 = np.linalg.norm(p - mic_locations[m2])
                tau_p = (dist1 - dist2) / c

                # Map delay to sample index (n_fft/2 is zero-delay center)
                sample_idx = int(tau_p * fs) + (n_fft // 2)

                # Accumulate correlation power if index is valid
                if 0 <= sample_idx < n_fft:
                    total_power += r_12[sample_idx]

            srp_map[i, j] = total_power

    # 4. Find location maximizing the Steered Response Power [cite: 112]
    max_idx = np.unravel_index(np.argmax(srp_map), srp_map.shape)
    estimated_pos = [x_range[max_idx[0]], y_range[max_idx[1]], z_fixed]

    return srp_map, estimated_pos, (x_range, y_range)


def plot_srp_map(srp_map, x_range, y_range, true_source_pos, estimated_pos, mic_locations):
    """
    Visualizes the SRP-PHAT power map with true and estimated source positions.
    """
    plt.figure(figsize=(10, 8))

    # Plot the likelihood map (transposed to match X/Y axes correctly)
    # Using 'inferno' or 'viridis' colormaps which are common for likelihood maps
    extent = [x_range[0], x_range[-1], y_range[0], y_range[-1]]
    im = plt.imshow(srp_map.T, extent=extent, origin='lower', cmap='inferno', aspect='auto')
    plt.colorbar(im, label='Likelihood (Normalized Power)')

    # Plot True Source Position (Red dot as seen in Fig 1 & 2 of the course)
    plt.scatter(true_source_pos[0], true_source_pos[1], color='red', s=100,
                label='Actual DOA/Location', edgecolors='white')

    # Plot Estimated Position (Black dot/triangle)
    plt.scatter(estimated_pos[0], estimated_pos[1], color='black', s=100, marker='x',
                label='Estimated Location (Map Max)')

    # Plot Microphone Locations (Blue dots)
    plt.scatter(mic_locations[:, 0], mic_locations[:, 1], color='cyan', marker='o',
                label='Microphones', edgecolors='black')

    plt.title(f"SRP-PHAT Power Map (Resolution: {srp_map.shape[0]}x{srp_map.shape[1]})")
    plt.xlabel("X-axis [m]")
    plt.ylabel("Y-axis [m]")
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.show()


def apply_srp_phat(mic_sigs, fs, room_dim, mic_locations, resolution, true_source_pos):
    f, t, signals_stft = apply_stft(mic_sigs, fs)
    gcc_channels = compute_gcc_phat(signals_stft)
    srp_map, estimated_pos, (x_range, y_range) = compute_srp_map(gcc_channels, fs, room_dim, mic_locations, resolution[0], resolution[1])
    plot_srp_map(srp_map, x_range, y_range, true_source_pos, estimated_pos, np.array(mic_locations))

