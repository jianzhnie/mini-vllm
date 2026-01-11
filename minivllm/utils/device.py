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
    available first. Supports cpu, cuda, xpu, npu, mps, mlu, musa. By default,
    it tries to use the best available accelerator device.

    Returns:
        Supported PyTorch device. Priority order:
        1. NPU (Huawei Ascend)
        2. CUDA (NVIDIA GPU)
        3. MUSA (Moonshot)
        4. MLU (Cambricon)
        5. XPU (Intel GPU)
        6. MPS (Apple Silicon)
        7. CPU (fallback)
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
    elif is_torch_mps_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device


def get_current_device(use_cpu: bool = False) -> torch.device:
    """Get the current process's device based on LOCAL_RANK environment variable.

    Uses the LOCAL_RANK environment variable to determine which device this
    process should use. Falls back to device 0 if LOCAL_RANK is not set.

    Note:
        MPS devices typically only support a single device, so the index
        is always 0 regardless of LOCAL_RANK.

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
        # MPS only supports a single device, always use index 0
        device = 'mps'
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
        Backend name: 'nccl' for CUDA, 'hccl' for NPU, 'ccl' for XPU,
        'gloo' for CPU and other devices.
    """
    if is_torch_npu_available():
        return 'hccl'  # Huawei NPU backend
    elif is_torch_cuda_available():
        return 'nccl'  # NVIDIA CUDA backend
    elif is_torch_xpu_available():
        return 'ccl'  # Intel XPU backend (oneCCL)
    elif is_torch_mlu_available():
        return 'cncl'  # Cambricon MLU backend
    elif is_torch_musa_available():
        return 'musa'  # Moonshot MUSA backend
    else:
        return 'gloo'  # Fallback to gloo for CPU, MPS and other devices


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

    Note:
        For devices that don't support memory querying (MPS, MLU, MUSA),
        returns default values. For CPU, attempts to use psutil if available.
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
        except (AttributeError, RuntimeError) as e:
            # Fallback: return a large value if API not available
            logger.warning(
                f'mem_get_info not available for NPU: {e}, returning default values'
            )
            return (10**12, 10**12)  # 1TB default
    elif device_type == 'xpu':
        try:
            return torch.xpu.mem_get_info(device)
        except (AttributeError, RuntimeError) as e:
            logger.warning(
                f'mem_get_info not available for XPU: {e}, returning default values'
            )
            return (10**12, 10**12)
    elif device_type == 'mps':
        # MPS doesn't have a direct memory query API
        # Try to get system memory as fallback
        try:
            import psutil

            # MPS uses unified memory, approximate with system memory
            total = psutil.virtual_memory().total
            # Assume 50% available for MPS (conservative estimate)
            free = total // 2
            return (free, total)
        except ImportError:
            logger.warning(
                'psutil not available, using default memory values for MPS')
            return (10**12, 10**12)  # 1TB default
    else:
        # For CPU and other devices, return system memory
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
        Dictionary of memory statistics. Returns empty dict for devices
        that don't support memory statistics (MPS, MLU, MUSA, CPU).
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
        except (AttributeError, RuntimeError) as e:
            logger.debug(f'memory_stats not available for NPU: {e}')
            return {}
    elif device_type == 'xpu':
        try:
            return torch.xpu.memory_stats(device)
        except (AttributeError, RuntimeError) as e:
            logger.debug(f'memory_stats not available for XPU: {e}')
            return {}
    elif device_type == 'mps':
        # MPS doesn't provide detailed memory statistics
        # Return basic info if available
        logger.debug(
            'memory_stats not available for MPS, returning empty dict')
        return {}
    else:
        # CPU and other devices don't have memory_stats
        return {}


def supports_device_graph() -> bool:
    """Check if the current device supports device graph optimization.

    Currently, only CUDA devices support graph optimization. Other devices
    may support similar optimizations in the future.

    Returns:
        True if device graph is supported, False otherwise.
    """
    # Only CUDA devices support graph optimization currently
    # Note: Other devices may have similar graph optimizations in the future
    return is_torch_cuda_available()


def supports_cuda_graph() -> bool:
    """Check if the current device supports CUDA Graph optimization.

    This is an alias for supports_device_graph() for backward compatibility.

    Returns:
        True if CUDA Graph is supported, False otherwise.
    """
    return supports_device_graph()


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


def should_use_pin_memory(device: Optional[torch.device] = None) -> bool:
    """Check if pin_memory should be used for the given device.

    pin_memory is primarily beneficial for CUDA devices to enable faster
    CPU-to-GPU transfers. For other devices, it may not be supported or
    may not provide benefits.

    Args:
        device: The device to check. If None, uses current device.

    Returns:
        True if pin_memory should be used, False otherwise.
    """
    if device is None:
        device = get_current_device()

    device_type = device.type
    # pin_memory is primarily useful for CUDA devices
    # Other devices may not support it or may not benefit from it
    return device_type == 'cuda'


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
            Note: non_blocking is only effective for CUDA devices.
            Other devices may ignore this parameter.

    Returns:
        The tensor on the target device.

    Raises:
        RuntimeError: If the device is not available or tensor movement fails.
    """
    try:
        return tensor.to(device, non_blocking=non_blocking)
    except RuntimeError as e:
        logger.error(f'Failed to move tensor to device {device}: {e}')
        raise RuntimeError(
            f'Cannot move tensor to device {device}. '
            f'Please ensure the device is available and accessible.') from e


def is_device_available(device: torch.device) -> bool:
    """Check if a device is available and accessible.

    Args:
        device: The device to check.

    Returns:
        True if the device is available, False otherwise.
    """
    device_type = device.type
    try:
        if device_type == 'cuda':
            return torch.cuda.is_available(
            ) and device.index is not None and device.index < torch.cuda.device_count(
            )
        elif device_type == 'npu':
            return is_torch_npu_available() and (
                device.index is None
                or device.index < torch.npu.device_count())
        elif device_type == 'xpu':
            return is_torch_xpu_available() and (
                device.index is None
                or device.index < torch.xpu.device_count())
        elif device_type == 'mps':
            return is_torch_mps_available(
            ) and torch.backends.mps.is_available()
        elif device_type == 'mlu':
            return is_torch_mlu_available() and (
                device.index is None or device.index < get_device_count())
        elif device_type == 'musa':
            return is_torch_musa_available() and (
                device.index is None or device.index < get_device_count())
        elif device_type == 'cpu':
            return True
        else:
            logger.warning(f'Unknown device type: {device_type}')
            return False
    except Exception as e:
        logger.warning(f'Error checking device availability for {device}: {e}')
        return False


def validate_device(device: torch.device) -> None:
    """Validate that a device is available and raise an error if not.

    Args:
        device: The device to validate.

    Raises:
        RuntimeError: If the device is not available.
    """
    if not is_device_available(device):
        raise RuntimeError(
            f'Device {device} is not available. '
            f'Please check that the device is properly installed and accessible.'
        )
