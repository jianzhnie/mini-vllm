"""Device management utilities for multi-device support.

Supports CUDA, NPU, XPU, MPS, MLU, and MUSA accelerators.
"""

import os
from typing import Any

import torch
from transformers.utils import (
    is_torch_cuda_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)

from minivllm.utils.logger_utils import get_logger

logger = get_logger(__name__)

# Device types with standard torch.{type} module and set_device/mem APIs.
_ACCELERATOR_TYPES = frozenset(('cuda', 'npu', 'xpu'))


def _get_device_type() -> str:
    """Detect best available device type (priority order)."""
    if is_torch_npu_available():
        return 'npu'
    if is_torch_cuda_available():
        return 'cuda'
    if is_torch_musa_available():
        return 'musa'
    if is_torch_mlu_available():
        return 'mlu'
    if is_torch_xpu_available():
        return 'xpu'
    if is_torch_mps_available():
        return 'mps'
    return 'cpu'


def get_visible_devices_keyword() -> str:
    """Get the environment variable keyword for visible devices."""
    if is_torch_cuda_available():
        return 'CUDA_VISIBLE_DEVICES'
    if is_torch_npu_available():
        return 'ASCEND_RT_VISIBLE_DEVICES'
    if is_torch_xpu_available():
        return 'XPU_VISIBLE_DEVICES'
    return ''


def get_dist_info() -> tuple[int, int, int]:
    """Get distributed training information: (rank, world_size, local_rank)."""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    return rank, world_size, local_rank


def get_current_device(use_cpu: bool = False) -> torch.device:
    """Get current process device based on LOCAL_RANK.

    Override via MINIVLLM_DEVICE env var (cpu, cuda, npu, xpu, mps, mlu, musa).
    MPS always uses index 0 (single device).
    """
    _, _, local_rank = get_dist_info()

    env_device = os.environ.get('MINIVLLM_DEVICE', '').lower().strip()
    if env_device:
        if env_device == 'cpu':
            return torch.device('cpu')
        return torch.device(f"{env_device}:{local_rank}")

    if use_cpu:
        return torch.device('cpu')

    dtype = _get_device_type()
    if dtype == 'mps':
        return torch.device('mps')
    if dtype == 'cpu':
        return torch.device('cpu')
    return torch.device(f"{dtype}:{local_rank}")


def get_device_count() -> int:
    """Number of available devices for current device type."""
    dtype = _get_device_type()
    if dtype == 'cpu':
        return 0
    module = getattr(torch, dtype, None)
    if module is None:
        return 0
    count_fn = getattr(module, 'device_count', None)
    if count_fn is None:
        return 0
    return count_fn()


def set_device(device: torch.device) -> None:
    """Set current device for the given device type."""
    device_type = device.type
    module = getattr(torch, device_type, None)
    set_fn = getattr(module, 'set_device', None) if module else None
    if set_fn is not None:
        try:
            set_fn(device)
        except Exception as e:
            logger.warning(f"Failed to set device {device}: {e}")


def get_default_device_name() -> str:
    """Device name string: 'cuda', 'npu', 'xpu', 'mps', 'mlu', 'musa', 'cpu'."""
    return _get_device_type()


def get_distributed_backend() -> str:
    """Appropriate distributed backend for current device."""
    dtype = _get_device_type()
    backends = {
        'npu': 'hccl',
        'cuda': 'nccl',
        'xpu': 'ccl',
        'mlu': 'cncl',
        'musa': 'musa',
    }
    return backends.get(dtype, 'gloo')


def empty_cache() -> None:
    """Free unused cached memory on current device."""
    dtype = _get_device_type()
    if dtype in _ACCELERATOR_TYPES:
        module = getattr(torch, dtype)
        fn = getattr(module, 'empty_cache', None)
        if fn is not None:
            fn()


def synchronize(device: torch.device | None = None) -> None:
    """Synchronize pending operations on device."""
    if device is None:
        device = get_current_device()
    if device.type in _ACCELERATOR_TYPES:
        module = getattr(torch, device.type)
        fn = getattr(module, 'synchronize', None)
        if fn is not None:
            fn(device)


def reset_peak_memory_stats(device: torch.device | None = None) -> None:
    """Reset peak memory statistics for device."""
    if device is None:
        device = get_current_device()
    if device.type in _ACCELERATOR_TYPES:
        module = getattr(torch, device.type)
        fn = getattr(module, 'reset_peak_memory_stats', None)
        if fn is not None:
            fn(device)


def mem_get_info(device: torch.device | None = None) -> tuple[int, int]:
    """Get (free_memory, total_memory) in bytes for device."""
    if device is None:
        device = get_current_device()

    device_type = device.type
    if device_type in _ACCELERATOR_TYPES:
        module = getattr(torch, device_type)
        fn = getattr(module, 'mem_get_info', None)
        if fn is not None:
            try:
                return fn(device)
            except RuntimeError:
                pass

    # CPU / MPS / fallback: use psutil for system memory
    try:
        import psutil

        total = psutil.virtual_memory().total
        free = psutil.virtual_memory().available
        return (free, total)
    except ImportError:
        logger.warning('psutil not available, using default memory values')
        return (10**12, 10**12)


def memory_stats(device: torch.device | None = None) -> dict[str, Any]:
    """Get memory statistics dict for device."""
    if device is None:
        device = get_current_device()
    if device.type in _ACCELERATOR_TYPES:
        module = getattr(torch, device.type)
        fn = getattr(module, 'memory_stats', None)
        if fn is not None:
            try:
                return fn(device)
            except RuntimeError:
                pass
    return {}


def supports_cuda_graph() -> bool:
    """Whether current device supports CUDA Graph optimization."""
    dtype = _get_device_type()
    return dtype in ('cuda', 'npu')


# Alias kept for backward compatibility
supports_device_graph = supports_cuda_graph


def get_device_capabilities(
        device: torch.device | None = None) -> dict[str, Any]:
    """Get capability dict for device."""
    if device is None:
        device = get_current_device()
    dt = device.type
    is_accel = dt in _ACCELERATOR_TYPES
    return {
        'device_type': dt,
        'supports_graph': supports_cuda_graph(),
        'supports_empty_cache': is_accel,
        'supports_synchronize': is_accel,
        'supports_memory_stats': is_accel,
    }


def should_use_pin_memory(device: torch.device | None = None) -> bool:
    """Whether pin_memory is beneficial for device."""
    if device is None:
        device = get_current_device()
    return device.type in ('cuda', 'npu')


def move_tensor_to_device(tensor: torch.Tensor,
                          device: torch.device,
                          non_blocking: bool = False) -> torch.Tensor:
    """Move tensor to device with consistent error handling."""
    try:
        return tensor.to(device, non_blocking=non_blocking)
    except RuntimeError as e:
        raise RuntimeError(f"Cannot move tensor to device {device}. "
                           f"Ensure the device is available.") from e


def is_device_available(device: torch.device) -> bool:
    """Check if a device is available and accessible."""
    dt = device.type
    if dt == 'cpu':
        return True
    module = getattr(torch, dt, None)
    if module is None:
        return False
    count_fn = getattr(module, 'device_count', None)
    if count_fn is None:
        return False
    try:
        count = count_fn()
    except Exception:
        return False
    if count == 0:
        return False
    return device.index is None or device.index < count


def validate_device(device: torch.device) -> None:
    """Raise RuntimeError if device is not available."""
    if not is_device_available(device):
        raise RuntimeError(f"Device {device} is not available. "
                           f"Check that the device is properly installed.")
