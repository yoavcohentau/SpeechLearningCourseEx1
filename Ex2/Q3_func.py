import torch
from denoiser.demucs import Demucs


def load_dns48_model(weights_path):
    model = Demucs(hidden=48)

    try:
        weights = torch.load(weights_path, map_location='cpu')
        if 'state_dict' in weights:
            model.load_state_dict(weights['state_dict'])
        elif 'model' in weights:
            model.load_state_dict(weights['model'])
        else:
            model.load_state_dict(weights)
        model.eval()
        return model

    except Exception as e:
        print(f"Error loading weights from {weights_path}: {e}")
        return None


def apply_deep_denoiser(noisy_signal_1d, model):
    # Convert to Tensor
    noisy_tensor = torch.from_numpy(noisy_signal_1d).float()

    # Add dimensions: (Time) -> (Batch=1, Channels=1, Time)
    if noisy_tensor.ndim == 1:
        noisy_tensor = noisy_tensor.unsqueeze(0).unsqueeze(0)

    # Inference
    with torch.no_grad():
        estimate = model(noisy_tensor)

    # reshape result: (1, 1, Time) -> (Time,)
    clean_signal = estimate.squeeze().numpy()

    # output length matches input length
    input_len = noisy_signal_1d.shape[0]
    clean_signal = clean_signal[:input_len]

    return clean_signal
