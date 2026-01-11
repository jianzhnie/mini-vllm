import os
from typing import Any, Dict, Optional, Tuple

import torch
from transformers.utils import (is_torch_cuda_available,
                                is_torch_mlu_available, is_torch_mps_available,
                                is_torch_musa_available,
                                is_torch_npu_available, is_torch_xpu_available)

from minivllm.utils.logger_utils import get_logger

logger = get_logger(__name__)

# Device type constants for easier reference and type checking
DEVICE_TYPE_CUDA = 'cuda'
DEVICE_TYPE_NPU = 'npu'
DEVICE_TYPE_XPU = 'xpu'
DEVICE_TYPE_MPS = 'mps'
DEVICE_TYPE_MLU = 'mlu'
DEVICE_TYPE_MUSA = 'musa'
DEVICE_TYPE_CPU = 'cpu'


def get_visible_devices_keyword() -> str:
    """Get the environment variable keyword for visible devices.

    Different device types use different environment variables to control
    which devices are visible to the process.

    Returns:
        Environment variable name for visible devices:
        - 'CUDA_VISIBLE_DEVICES' for CUDA devices
        - 'ASCEND_RT_VISIBLE_DEVICES' for NPU devices
        - 'XPU_VISIBLE_DEVICES' for XPU devices (if supported)
        - Empty string for other device types
    """
    if is_torch_cuda_available():
        return 'CUDA_VISIBLE_DEVICES'
    elif is_torch_npu_available():
        return 'ASCEND_RT_VISIBLE_DEVICES'
    elif is_torch_xpu_available():
        # Intel XPU may use different env var, adjust if needed
        return 'XPU_VISIBLE_DEVICES'
    else:
        return ''


def get_dist_info() -> Tuple[int, int, int]:
    """Get distributed training information.

    Returns:
        Tuple of (rank, world_size, local_rank)
    """
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    return rank, world_size, local_rank


def get_device() -> torch.device:
    """Retrieve PyTorch device. It checks that the requested device is
    available first. For now, it supports cpu and cuda, xpu, npu. By default,
    it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    if is_torch_npu_available():
        device = torch.device('npu:0')
        torch.npu.set_device(device)
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    elif is_torch_musa_available():
        device = torch.device('musa:0')
    elif is_torch_mlu_available():
        device = torch.device('mlu:0')
    elif is_torch_xpu_available():
        device = torch.device('xpu:0')
        torch.xpu.set_device(device)
    else:
        device = torch.device('cpu')
    return device


def get_current_device(use_cpu: bool = False) -> torch.device:
    """Get the current process's device based on LOCAL_RANK environment variable.

    Uses the LOCAL_RANK environment variable to determine which device this
    process should use. Falls back to device 0 if LOCAL_RANK is not set.

    Returns:
        torch.device: Current process's device
    """
    _, _, local_rank = get_dist_info()

    if use_cpu:
        device = 'cpu'
    elif is_torch_cuda_available():
        device = 'cuda:{}'.format(local_rank)
    elif is_torch_npu_available():
        device = 'npu:{}'.format(local_rank)
    elif is_torch_xpu_available():
        device = 'xpu:{}'.format(local_rank)
    elif is_torch_mps_available():
        device = 'mps:{}'.format(local_rank)
    elif is_torch_mlu_available():
        device = 'mlu:{}'.format(local_rank)
    elif is_torch_musa_available():
        device = 'musa:{}'.format(local_rank)
    else:
        device = 'cpu'
    return torch.device(device)


def get_device_count() -> int:
    """Get the number of available devices for the current device type.

    Returns:
        Number of available devices. Returns 0 for CPU or unsupported devices.
    """
    if is_torch_npu_available():
        num_devices = torch.npu.device_count()
    elif is_torch_xpu_available():
        num_devices = torch.xpu.device_count()
    elif is_torch_cuda_available():
        num_devices = torch.cuda.device_count()
    elif is_torch_mps_available():
        # MPS typically supports only one device
        num_devices = 1 if torch.backends.mps.is_available() else 0
    elif is_torch_mlu_available():
        try:
            num_devices = torch.mlu.device_count()
        except AttributeError:
            num_devices = 0
    elif is_torch_musa_available():
        try:
            num_devices = torch.musa.device_count()
        except AttributeError:
            num_devices = 0
    else:
        num_devices = 0
    return num_devices


def set_device(device: torch.device) -> None:
    """Set the current device for the given device type.

    Args:
        device: The device to set as current.

    Note:
        Some device types (MPS, MLU, MUSA) don't have set_device methods
        and will be silently ignored.
    """
    device_type = device.type
    try:
        if device_type == 'cuda':
            torch.cuda.set_device(device)
        elif device_type == 'npu':
            torch.npu.set_device(device)
        elif device_type == 'xpu':
            torch.xpu.set_device(device)
        # Other device types (mps, mlu, musa) don't have set_device methods
        # and are handled automatically by PyTorch
    except Exception as e:
        logger.warning(
            f'Failed to set device {device}: {e}. '
            f'This may be normal for device types that don\'t support set_device.'
        )


def get_default_device_name() -> str:
    """Get the default device name string based on available devices.

    Returns:
        Device name string like 'cuda', 'npu', 'xpu', etc.
    """
    if is_torch_npu_available():
        return 'npu'
    elif is_torch_cuda_available():
        return 'cuda'
    elif is_torch_xpu_available():
        return 'xpu'
    elif is_torch_mps_available():
        return 'mps'
    elif is_torch_mlu_available():
        return 'mlu'
    elif is_torch_musa_available():
        return 'musa'
    else:
        return 'cpu'


def get_distributed_backend() -> str:
    """Get the appropriate distributed backend for the current device.

    Returns:
        Backend name: 'nccl' for CUDA, 'hccl' for NPU, 'gloo' for CPU, etc.
    """
    if is_torch_npu_available():
        return 'hccl'
    elif is_torch_cuda_available():
        return 'nccl'
    elif is_torch_xpu_available():
        return 'ccl'  # Intel oneCCL
    else:
        return 'gloo'  # Fallback to gloo for CPU and other devices


def empty_cache() -> None:
    """Empty the cache for the current device.

    This function attempts to free unused memory cached by the device.
    Not all device types support this operation, in which case it will
    be silently ignored.
    """
    try:
        if is_torch_npu_available():
            torch.npu.empty_cache()
        elif is_torch_cuda_available():
            torch.cuda.empty_cache()
        elif is_torch_xpu_available():
            torch.xpu.empty_cache()
        # Other devices (MPS, MLU, MUSA, CPU) don't have empty_cache methods
        # or don't need explicit cache clearing
    except Exception as e:
        logger.debug(
            f'Failed to empty cache: {e}. This may be normal for some device types.'
        )


def synchronize(device: Optional[torch.device] = None) -> None:
    """Synchronize operations on the given device.

    This ensures all pending operations on the device are completed.
    For devices that don't support synchronization (MPS, CPU), this is a no-op.

    Args:
        device: The device to synchronize. If None, uses current device.
    """
    if device is None:
        device = get_current_device()

    device_type = device.type
    try:
        if device_type == 'cuda':
            torch.cuda.synchronize(device)
        elif device_type == 'npu':
            torch.npu.synchronize(device)
        elif device_type == 'xpu':
            torch.xpu.synchronize(device)
        # Other devices (MPS, MLU, MUSA, CPU) don't have synchronize methods
        # or don't need explicit synchronization
    except Exception as e:
        logger.debug(
            f'Failed to synchronize device {device}: {e}. '
            f'This may be normal for devices that don\'t support synchronization.'
        )


def reset_peak_memory_stats(device: Optional[torch.device] = None) -> None:
    """Reset peak memory statistics for the given device.

    Args:
        device: The device to reset stats for. If None, uses current device.
    """
    if device is None:
        device = get_current_device()

    device_type = device.type
    if device_type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    elif device_type == 'npu':
        # NPU may not have reset_peak_memory_stats, skip if not available
        try:
            torch.npu.reset_peak_memory_stats(device)
        except AttributeError:
            pass
    elif device_type == 'xpu':
        try:
            torch.xpu.reset_peak_memory_stats(device)
        except AttributeError:
            pass
    # Other devices may not support this


def mem_get_info(device: Optional[torch.device] = None) -> Tuple[int, int]:
    """Get memory information (free, total) for the given device.

    Args:
        device: The device to query. If None, uses current device.

    Returns:
        Tuple of (free_memory, total_memory) in bytes.
    """
    if device is None:
        device = get_current_device()

    device_type = device.type
    if device_type == 'cuda':
        return torch.cuda.mem_get_info(device)
    elif device_type == 'npu':
        # NPU uses different API
        try:
            return torch.npu.mem_get_info(device)
        except AttributeError:
            # Fallback: return a large value if API not available
            logger.warning(
                'mem_get_info not available for NPU, returning default values')
            return (10**12, 10**12)  # 1TB default
    elif device_type == 'xpu':
        try:
            return torch.xpu.mem_get_info(device)
        except AttributeError:
            logger.warning(
                'mem_get_info not available for XPU, returning default values')
            return (10**12, 10**12)
    else:
        # For CPU and other devices, return a large default value
        try:
            import psutil
            return (psutil.virtual_memory().available,
                    psutil.virtual_memory().total)
        except ImportError:
            # If psutil is not available, return a large default value
            logger.warning(
                'psutil not available, using default memory values for CPU')
            return (10**12, 10**12)  # 1TB default


def memory_stats(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Get memory statistics for the given device.

    Args:
        device: The device to query. If None, uses current device.

    Returns:
        Dictionary of memory statistics.
    """
    if device is None:
        device = get_current_device()

    device_type = device.type
    if device_type == 'cuda':
        return torch.cuda.memory_stats(device)
    elif device_type == 'npu':
        # NPU may not have memory_stats, return empty dict
        try:
            return torch.npu.memory_stats(device)
        except AttributeError:
            logger.warning('memory_stats not available for NPU')
            return {}
    elif device_type == 'xpu':
        try:
            return torch.xpu.memory_stats(device)
        except AttributeError:
            logger.warning('memory_stats not available for XPU')
            return {}
    else:
        return {}


def supports_cuda_graph() -> bool:
    """Check if the current device supports CUDA Graph optimization.

    Currently, only CUDA devices support graph optimization. Other devices
    may support similar optimizations in the future.

    Returns:
        True if CUDA Graph is supported, False otherwise.
    """
    # Only CUDA devices support CUDA Graph currently
    # Note: Other devices may have similar graph optimizations in the future
    return is_torch_cuda_available()


def get_device_capabilities(
        device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Get capabilities and features supported by the device.

    This function returns a dictionary describing what features are available
    on the given device, such as graph optimization, memory management, etc.

    Args:
        device: The device to query. If None, uses current device.

    Returns:
        Dictionary with device capabilities:
        - 'supports_graph': Whether device supports graph optimization
        - 'supports_empty_cache': Whether device supports cache clearing
        - 'supports_synchronize': Whether device supports synchronization
        - 'supports_memory_stats': Whether device supports memory statistics
        - 'device_type': Type of device ('cuda', 'npu', 'xpu', etc.)
    """
    if device is None:
        device = get_current_device()

    device_type = device.type
    capabilities = {
        'device_type': device_type,
        'supports_graph': device_type == 'cuda',
        'supports_empty_cache': device_type in ('cuda', 'npu', 'xpu'),
        'supports_synchronize': device_type in ('cuda', 'npu', 'xpu'),
        'supports_memory_stats': device_type in ('cuda', 'npu', 'xpu'),
    }

    return capabilities


def move_tensor_to_device(tensor: torch.Tensor,
                          device: torch.device,
                          non_blocking: bool = False) -> torch.Tensor:
    """Move a tensor to the specified device.

    This is a device-agnostic wrapper around tensor.to() that handles
    pin_memory and non_blocking appropriately.

    Args:
        tensor: The tensor to move.
        device: The target device.
        non_blocking: Whether to use non-blocking transfer.

    Returns:
        The tensor on the target device.
    """
    return tensor.to(device, non_blocking=non_blocking)
