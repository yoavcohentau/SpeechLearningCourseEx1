import torch
import numpy as np
from denoiser.demucs import Demucs


def load_dns48_model(weights_path):
    """
    Loads the DNS48 model architecture and loads the local weights file.
    """
    # Initialize the Demucs model with DNS48 parameters
    # DNS48 implies hidden=48. Other parameters are usually defaults.
    model = Demucs(hidden=48)

    # Load the weights from the local file
    try:
        # Load package
        pkg = torch.load(weights_path, map_location='cpu')

        # Determine if it's a full checkpoint or just state_dict
        if 'state_dict' in pkg:
            model.load_state_dict(pkg['state_dict'])
        elif 'model' in pkg:
            model.load_state_dict(pkg['model'])
        else:
            # Assume the file is the state_dict itself
            model.load_state_dict(pkg)

        model.eval()
        return model

    except Exception as e:
        print(f"Error loading weights from {weights_path}: {e}")
        return None


def apply_deep_denoiser(noisy_signal_1d, model, fs=16000):
    """
    Applies the Deep Learning Denoiser to a single channel signal.
    Args:
        noisy_signal_1d: Numpy array (Time,)
        model: Loaded PyTorch model
    Returns:
        Cleaned signal (Numpy array)
    """
    # 1. Convert to Tensor
    noisy_tensor = torch.from_numpy(noisy_signal_1d).float()

    # 2. Add dimensions: (Time) -> (Batch=1, Channels=1, Time)
    if noisy_tensor.ndim == 1:
        noisy_tensor = noisy_tensor.unsqueeze(0).unsqueeze(0)

    # 3. Inference
    with torch.no_grad():
        # Demucs expects input shape (Batch, Channels, Time)
        # It handles normalization internally usually, but we pass raw audio.
        estimate = model(noisy_tensor)

    # 4. Extract result: (1, 1, Time) -> (Time,)
    clean_signal = estimate.squeeze().numpy()

    # Ensure output length matches input length (Model might pad)
    input_len = noisy_signal_1d.shape[0]
    clean_signal = clean_signal[:input_len]

    return clean_signal