import matplotlib.pyplot as plt
import numpy as np
import rir_generator as rir
from scipy import signal
from scipy.signal import fftconvolve


# ---Room Simulation---
def generate_room_impulse_responses(fs, room_dim, mic_locations, source_location, T60_values):
    c = 344.0  # speed of sound
    all_rirs = {}
    for t60 in T60_values:
        nsample = int(t60 * fs)

        # Generate RIRs
        h = rir.generate(
            c=c,
            fs=fs,
            r=np.array(mic_locations),  # Mic array coordinates
            s=source_location,  # Source position 'p'
            L=np.array(room_dim),  # Dimensions of room
            reverberation_time=t60,  # T60 for simulation
            nsample=nsample  # Total number of RIR samples
        )
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


def mix_signals(clean_signal, noise_signal, snr_db):
    # Calculate power of clean signal and noise
    p_signal = np.mean(clean_signal ** 2)
    p_noise = np.mean(noise_signal ** 2)

    # Avoid division by zero
    if p_noise == 0:
        return clean_signal

    # Calculate required scaling factor for noise
    # SNR_linear = P_signal / (scale^2 * P_noise)
    # scale = sqrt(P_signal / (P_noise * 10^(SNR/10)))
    scale_factor = np.sqrt(p_signal / (p_noise * (10 ** (snr_db / 10))))

    # Create noisy signal
    noise = scale_factor * noise_signal
    noisy_signal = clean_signal + noise

    return noisy_signal, noise


def generate_white_noise(signal_shape):
    # generate normal distribution samples (mean=0, std=1)
    noise = np.random.randn(*signal_shape)
    return noise


# ---utils---
def apply_stft(signals, fs, nperseg=512, noverlap=256):
    stft_data = []
    for i in range(signals.shape[0]):
        f, t, Z = signal.stft(signals[i], fs=fs, window='hamming', nperseg=nperseg, noverlap=noverlap)
        stft_data.append(Z)
    return f, t, np.array(stft_data)
    #     stft_data.append(Z[16:81,:])
    # return f[16:81], t, np.array(stft_data)


def plot_location_maps(maps, method_names, x_range, y_range, true_source_pos, estimated_positions, mic_locations):
    # Ensure inputs are lists type
    if not isinstance(maps, list):
        maps = [maps]
        method_names = [method_names]
        estimated_positions = [estimated_positions]

    num_maps = len(maps)
    fig, axes = plt.subplots(1, num_maps, figsize=(5 * num_maps, 4), squeeze=False)

    extent = [x_range[0], x_range[-1], y_range[0], y_range[-1]]
    mic_locations = np.array(mic_locations)

    for i in range(num_maps):
        ax = axes[0, i]
        curr_map = maps[i]
        curr_method = method_names[i]
        curr_est = estimated_positions[i]
        true_pos = np.array(true_source_pos)

        error_m = np.linalg.norm(true_pos[:2] - curr_est[:2])

        # Plot the likelihood map (Pseudo-spectrum for MUSIC or Power Map for SRP)
        im = ax.imshow(curr_map.T, extent=extent, origin='lower', cmap='inferno', aspect='auto')
        fig.colorbar(im, ax=ax, label='Likelihood / Power')

        # Plot True Source Position (Red dot)
        ax.scatter(true_source_pos[0], true_source_pos[1], color='red', s=100,
                   label='True Source', edgecolors='white', zorder=5)

        # Plot Estimated Position (Black X)
        ax.scatter(curr_est[0], curr_est[1], color='black', s=150, marker='x',
                   label=f'Est. {curr_method}', zorder=6)

        # Plot Microphone Locations (dots)
        ax.scatter(mic_locations[:, 0], mic_locations[:, 1], color='cyan', marker='o',
                   label='Microphones', edgecolors='black', zorder=4)

        ax.set_title(f"{curr_method} Map ({curr_map.shape[0]}x{curr_map.shape[1]})\nError: {error_m:.3f} meters")
        ax.set_xlabel("X-axis [m]")
        ax.set_ylabel("Y-axis [m]")
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def calculate_rmse(true_positions, estimated_positions):
    true_pos = np.array(true_positions)
    est_pos = np.array(estimated_positions)

    squared_distances = np.sum((true_pos - est_pos) ** 2, axis=1)
    mse = np.mean(squared_distances)
    rmse = np.sqrt(mse)

    return rmse


def plot_rmse_performance(experiment_results, x_axis_type="SNR"):
    """
    Plots RMSE performance for SRP-PHAT and MUSIC side-by-side.

    Args:
        experiment_results (list): List containing dictionaries from q2_func runs.
        x_axis_type (list): List of strings ['SNR', 'T60'] defining the X-axis for each plot.
    """
    num_exps = len(experiment_results)
    # Create subplots based on number of experiments
    fig, axes = plt.subplots(1, num_exps, figsize=(6 * num_exps, 5), squeeze=False)

    for i, res in enumerate(experiment_results):
        ax = axes[0, i]
        current_type = x_axis_type[i]

        x_vals = []
        rmse_srp = []
        rmse_music = []

        # Parse the keys (e.g., "300ms-15db") based on requested plot type
        for key, vals in res.items():
            if current_type == "SNR":
                # Extract SNR value from string "XXXms-YYdb"
                val = int(key.split("-")[1].replace("db", ""))
                title_info = "RT = 0.3s"  # Fixed RT for SNR experiment
            else:
                # Extract T60 value from string "XXXms-YYdb"
                val = float(key.split("-")[0].replace("ms", ""))
                title_info = "SNR = 15dB"  # Fixed SNR for T60 experiment

            x_vals.append(val)
            rmse_srp.append(vals["rmse_srp_phat"])
            rmse_music.append(vals["rmse_music"])

        # Sort data points to ensure lines are plotted correctly
        x_vals, rmse_srp, rmse_music = zip(*sorted(zip(x_vals, rmse_srp, rmse_music)))

        # Plot SRP-PHAT (usually markers 'o') vs MUSIC (usually markers 's') [cite: 668]
        ax.plot(x_vals, rmse_srp, marker='o', linestyle='--', label="SRP-PHAT")
        ax.plot(x_vals, rmse_music, marker='s', linestyle='-', label="MUSIC")

        ax.set_xlabel(f"{current_type} [{'dB' if current_type == 'SNR' else 'ms'}]")
        ax.set_ylabel("RMSE [m]")
        ax.set_title(f"RMSE vs {current_type}\n({title_info})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


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


def compute_srp_map(gcc_channels, fs, room_dim, mic_locations, num_px_x=20, num_px_y=20, z=1.5):
    c = 344.0  # Speed of sound
    n_fft = len(next(iter(gcc_channels.values())))  # Get FFT length

    x_range = np.linspace(0, room_dim[0], num_px_x)
    y_range = np.linspace(0, room_dim[1], num_px_y)
    z_fixed = z

    srp_map = np.zeros((num_px_x, num_px_y))
    # for loop on pixel coordinate
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            p = np.array([x, y, z_fixed])
            total_power = 0

            # Sum the GCC-PHAT values for all microphone pairs
            for (m1, m2), r_12 in gcc_channels.items():
                # Calculate TDOA
                dist1 = np.linalg.norm(p - mic_locations[m1])
                dist2 = np.linalg.norm(p - mic_locations[m2])
                tau_p = (dist1 - dist2) / c

                # Map delay to sample index (n_fft/2 is zero-delay center)
                sample_idx = int(tau_p * fs) + (n_fft // 2)

                # Accumulate correlation power if index is valid
                if 0 <= sample_idx < n_fft:
                    total_power += r_12[sample_idx]

            srp_map[i, j] = total_power

    # Find max value on map: the estimated-position
    max_idx = np.unravel_index(np.argmax(srp_map), srp_map.shape)
    estimated_pos = [x_range[max_idx[0]], y_range[max_idx[1]], z_fixed]

    return srp_map, estimated_pos, (x_range, y_range)


def apply_srp_phat(mic_sigs, fs, room_dim, mic_locations, resolution, true_source_pos, plot_map=True):
    f, t, signals_stft = apply_stft(mic_sigs, fs)
    gcc_channels = compute_gcc_phat(signals_stft)
    srp_map, estimated_pos, (x_range, y_range) = compute_srp_map(gcc_channels, fs, room_dim, mic_locations,
                                                                 resolution[0], resolution[1])
    if plot_map:
        plot_location_maps(srp_map, "SRP-PHAT", x_range, y_range, true_source_pos, estimated_pos, mic_locations)

    return srp_map, estimated_pos, (x_range, y_range)


# ---MUSIC---
def estimate_cov_matrix(signals_stft):
    num_mics, num_freqs, num_frames = signals_stft.shape

    cov_matrix = np.zeros((num_freqs, num_mics, num_mics), dtype=complex)
    # calc estimated cov matrix for each f
    for f in range(num_freqs):
        x_f = signals_stft[:, f, :]
        cov_matrix[f] = np.dot(x_f, x_f.conj().T) / num_frames

    return cov_matrix


def find_noise_subspace(cov_matrix, num_sources=1):
    num_freqs, num_mics, _ = cov_matrix.shape
    num_noise_vectors = num_mics - num_sources

    U_N = np.zeros((num_freqs, num_mics, num_noise_vectors), dtype=complex)
    for f in range(num_freqs):
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix[f])

        # sort eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # The first 'num_sources' eigenvectors span the signal subspace (U_S) and
        # the other M-P eigenvectors span the noise subspace (U_N)
        U_N[f] = eigenvectors[:, num_sources:]

    return U_N


def compute_music_map(U_N, fs, room_dim, mic_locations, num_px_x=20, num_px_y=20, z=1.5):
    c = 344.0  # Speed of sound
    num_freqs, num_mics, num_noise_vectors = U_N.shape

    x_range = np.linspace(0, room_dim[0], num_px_x)
    y_range = np.linspace(0, room_dim[1], num_px_y)
    z_fixed = z

    music_map = np.zeros((num_px_x, num_px_y))
    freq_bins = np.linspace(0, fs / 2, num_freqs)
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            p = np.array([x, y, z_fixed])

            dists = np.linalg.norm(mic_locations - p, axis=1)
            taus = dists / c

            p_score = 0
            for f_idx, f_hz in enumerate(freq_bins):
                # find steering vector for location p
                steering_vector = np.exp(-1j * 2 * np.pi * f_hz * taus)

                # U_N_f is the matrix of eigenvectors spanning the noise subspace
                U_N_f = U_N[f_idx]

                # degree of orthogonality: e^H * U_N * U_N^H * e
                projection = np.matmul(steering_vector.conj().T, U_N_f)
                orthogonality_degree = np.sum(np.abs(projection) ** 2)
                p_score += 1.0 / (orthogonality_degree + 1e-10)  # pseudo-spectrum

            music_map[i, j] = p_score

    max_idx = np.unravel_index(np.argmax(music_map), music_map.shape)
    estimated_pos = [x_range[max_idx[0]], y_range[max_idx[1]], z_fixed]

    return music_map, estimated_pos, (x_range, y_range)


def apply_music(mic_sigs, fs, room_dim, mic_locations, resolution, true_source_pos, plot_map=True):
    f, t, signals_stft = apply_stft(mic_sigs, fs)
    cov_matrix = estimate_cov_matrix(signals_stft)
    U_N = find_noise_subspace(cov_matrix, 1)
    music_map, estimated_pos, (x_range, y_range) = compute_music_map(U_N, fs, room_dim, mic_locations,
                                                                     resolution[0], resolution[1])
    if plot_map:
        plot_location_maps(music_map, "MUSIC", x_range, y_range, true_source_pos, estimated_pos, mic_locations)

    return music_map, estimated_pos, (x_range, y_range)
