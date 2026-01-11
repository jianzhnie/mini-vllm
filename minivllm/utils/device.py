import os
from typing import Any, Dict, Optional, Tuple

import torch
from transformers.utils import (is_torch_cuda_available,
                                is_torch_mlu_available, is_torch_mps_available,
                                is_torch_musa_available,
                                is_torch_npu_available, is_torch_xpu_available)

from minivllm.utils.logger_utils import get_logger

logger = get_logger(__name__)


def get_visible_devices_keyword() -> str:
    """Function that gets visible devices keyword name.
    Returns:
        'CUDA_VISIBLE_DEVICES' or `ASCEND_RT_VISIBLE_DEVICES`
    """
    return 'CUDA_VISIBLE_DEVICES' if is_torch_cuda_available else 'ASCEND_RT_VISIBLE_DEVICES'


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
    r"""
    Gets the number of available GPU or NPU devices.
    """
    if is_torch_npu_available():
        num_devices = torch.npu.device_count()
    elif is_torch_xpu_available():
        num_devices = torch.xpu.device_count()
    elif is_torch_cuda_available():
        num_devices = torch.cuda.device_count()
    else:
        num_devices = 0
    return num_devices


def set_device(device: torch.device) -> None:
    """Set the current device for the given device type.

    Args:
        device: The device to set as current.
    """
    device_type = device.type
    if device_type == 'cuda':
        torch.cuda.set_device(device)
    elif device_type == 'npu':
        torch.npu.set_device(device)
    elif device_type == 'xpu':
        torch.xpu.set_device(device)
    # Other device types (mps, mlu, musa) don't have set_device methods


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
    """Empty the cache for the current device."""
    if is_torch_npu_available():
        torch.npu.empty_cache()
    elif is_torch_cuda_available():
        torch.cuda.empty_cache()
    elif is_torch_xpu_available():
        torch.xpu.empty_cache()
    # Other devices don't have empty_cache methods


def synchronize(device: Optional[torch.device] = None) -> None:
    """Synchronize operations on the given device.

    Args:
        device: The device to synchronize. If None, uses current device.
    """
    if device is None:
        device = get_current_device()

    device_type = device.type
    if device_type == 'cuda':
        torch.cuda.synchronize(device)
    elif device_type == 'npu':
        torch.npu.synchronize(device)
    elif device_type == 'xpu':
        torch.xpu.synchronize(device)
    # Other devices don't have synchronize methods or don't need it


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

    Returns:
        True if CUDA Graph is supported, False otherwise.
    """
    # Only CUDA devices support CUDA Graph
    return is_torch_cuda_available()


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
