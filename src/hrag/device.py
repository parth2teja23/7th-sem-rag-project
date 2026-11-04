def pick_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"
