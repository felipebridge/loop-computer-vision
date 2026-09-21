from __future__ import annotations

from traffic_intelligence.config.settings import DeviceType


def resolve_device(requested: DeviceType) -> str:
    if requested == DeviceType.CPU:
        return "cpu"
    if requested == DeviceType.CUDA:
        return "cuda"
    if requested == DeviceType.MPS:
        return "mps"

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        return "cpu"

