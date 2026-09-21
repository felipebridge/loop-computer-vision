from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from traffic_intelligence.config.settings import DeviceType
from traffic_intelligence.utils.device import resolve_device


def test_resolve_device_explicit_cpu():
    assert resolve_device(DeviceType.CPU) == "cpu"


def test_resolve_device_explicit_cuda():
    assert resolve_device(DeviceType.CUDA) == "cuda"


def test_resolve_device_explicit_mps():
    assert resolve_device(DeviceType.MPS) == "mps"


def test_resolve_device_auto_prefers_cuda_when_available():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    with patch.dict(sys.modules, {"torch": mock_torch}):
        assert resolve_device(DeviceType.AUTO) == "cuda"


def test_resolve_device_auto_selects_mps_when_cuda_unavailable():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = True

    with patch.dict(sys.modules, {"torch": mock_torch}):
        assert resolve_device(DeviceType.AUTO) == "mps"


def test_resolve_device_auto_falls_back_to_cpu_when_no_accelerator():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False

    with patch.dict(sys.modules, {"torch": mock_torch}):
        assert resolve_device(DeviceType.AUTO) == "cpu"


def test_resolve_device_auto_falls_back_to_cpu_when_torch_missing():
    with patch.dict(sys.modules, {"torch": None}):
        assert resolve_device(DeviceType.AUTO) == "cpu"
