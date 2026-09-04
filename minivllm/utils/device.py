"""Device management utilities for multi-device support.

Supports CUDA, NPU, XPU, MPS, MLU, and MUSA accelerators.
"""

import functools
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

# `is_torch_npu_available` is re-exported via the import at the top of this
# module so callers can use `from minivllm.utils.device import is_torch_npu_available`.

# Device types with standard torch.{type} module and set_device/mem APIs.
ACCELERATOR_TYPES = frozenset(("cuda", "npu", "xpu", "mlu", "musa"))

# Selection priority. NPU is checked first (matches this project's primary
# target hardware); each backend is gated on actually having a device present,
# so a CUDA-only box with `torch_npu` installed falls through to CUDA.
DEVICE_PRIORITY = ("npu", "cuda", "musa", "mlu", "xpu", "mps")
AVAILABILITY_CHECKS = {
    "npu": is_torch_npu_available,
    "cuda": is_torch_cuda_available,
    "musa": is_torch_musa_available,
    "mlu": is_torch_mlu_available,
    "xpu": is_torch_xpu_available,
    "mps": is_torch_mps_available,
}


def dtype_has_device(dtype: str) -> bool:
    """True if backend `dtype` reports at least one device.

    `is_torch_*_available()` only checks software (importability) for NPU, so
    we additionally require a non-zero device count to avoid targeting an empty
    accelerator. Backends without a `device_count` API fall back to trusting
    the availability check.
    """
    module = getattr(torch, dtype, None)
    count_fn = getattr(module, "device_count", None) if module else None
    if count_fn is None:
        return True
    try:
        return count_fn() > 0
    except Exception:
        return False


@functools.cache
def get_device_type() -> str:
    """Detect best available device type (priority order, hardware-gated).

    Honors MINIVLLM_DEVICE so capability gating (cuda graphs, dist backend,
    accelerator-specific ops) stays consistent with get_current_device.
    """
    env_device = os.environ.get("MINIVLLM_DEVICE", "").lower().strip()
    if env_device:
        return env_device
    for device_type in DEVICE_PRIORITY:
        if AVAILABILITY_CHECKS[device_type]() and dtype_has_device(device_type):
            return device_type
    return "cpu"


def get_visible_devices_keyword() -> str:
    """Get the environment variable keyword for visible devices.

    Derived from the same priority order as `get_device_type` so the two can
    never disagree about which accelerator is active.
    """
    known_keywords = {
        "npu": "ASCEND_RT_VISIBLE_DEVICES",
        "cuda": "CUDA_VISIBLE_DEVICES",
        "xpu": "XPU_VISIBLE_DEVICES",
    }
    return known_keywords.get(get_device_type(), "")


def get_dist_info() -> tuple[int, int, int]:
    """Get distributed training information: (rank, world_size, local_rank)."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world_size, local_rank


def get_current_device(use_cpu: bool = False) -> torch.device:
    """Get current process device based on LOCAL_RANK.

    Override via MINIVLLM_DEVICE env var (cpu, cuda, npu, xpu, mps, mlu, musa).
    MPS always uses index 0 (single device).
    """
    _, _, local_rank = get_dist_info()

    env_device = os.environ.get("MINIVLLM_DEVICE", "").lower().strip()
    if env_device:
        if env_device == "cpu":
            return torch.device("cpu")
        return torch.device(f"{env_device}:{local_rank}")

    if use_cpu:
        return torch.device("cpu")

    dtype = get_device_type()
    if dtype == "mps":
        return torch.device("mps")
    if dtype == "cpu":
        return torch.device("cpu")
    return torch.device(f"{dtype}:{local_rank}")


def get_device_count() -> int:
    """Number of available devices for current device type."""
    dtype = get_device_type()
    if dtype == "cpu":
        return 0
    module = getattr(torch, dtype, None)
    if module is None:
        return 0
    count_fn = getattr(module, "device_count", None)
    if count_fn is None:
        return 0
    return count_fn()


def set_device(device: torch.device) -> None:
    """Set current device for the given device type."""
    device_type = device.type
    module = getattr(torch, device_type, None)
    set_fn = getattr(module, "set_device", None) if module else None
    if set_fn is not None:
        try:
            set_fn(device)
        except Exception as e:
            # A worker that silently lands on the wrong device corrupts
            # tensor parallelism — fail loudly instead of warning.
            raise RuntimeError(f"Failed to set device {device}: {e}") from e


def get_default_device_name() -> str:
    """Device name string: 'cuda', 'npu', 'xpu', 'mps', 'mlu', 'musa', 'cpu'."""
    return get_device_type()


def get_distributed_backend() -> str:
    """Appropriate distributed backend for current device.

    Set MINIVLLM_TP_BACKEND to override (e.g. 'gloo' for CPU-based TP testing
    when HCCL P2P networking is unavailable on NPU).
    """
    import os

    override = os.environ.get("MINIVLLM_TP_BACKEND", "").lower()
    if override:
        import torch.distributed as dist

        if dist.is_available() and hasattr(dist, "Backend"):
            return override
    dtype = get_device_type()
    backends = {
        "npu": "hccl",
        "cuda": "nccl",
        "xpu": "ccl",
        "mlu": "cncl",
        "musa": "musa",
    }
    return backends.get(dtype, "gloo")


def empty_cache() -> None:
    """Free unused cached memory on current device."""
    dtype = get_device_type()
    if dtype in ACCELERATOR_TYPES:
        module = getattr(torch, dtype)
        fn = getattr(module, "empty_cache", None)
        if fn is not None:
            fn()


def synchronize(device: torch.device | None = None) -> None:
    """Synchronize pending operations on device."""
    if device is None:
        device = get_current_device()
    if device.type in ACCELERATOR_TYPES:
        module = getattr(torch, device.type)
        fn = getattr(module, "synchronize", None)
        if fn is not None:
            fn(device)


def reset_peak_memory_stats(device: torch.device | None = None) -> None:
    """Reset peak memory statistics for device."""
    if device is None:
        device = get_current_device()
    if device.type in ACCELERATOR_TYPES:
        module = getattr(torch, device.type)
        fn = getattr(module, "reset_peak_memory_stats", None)
        if fn is not None:
            fn(device)


def mem_get_info(device: torch.device | None = None) -> tuple[int, int]:
    """Get (free_memory, total_memory) in bytes for device."""
    if device is None:
        device = get_current_device()

    device_type = device.type
    if device_type in ACCELERATOR_TYPES:
        module = getattr(torch, device_type)
        fn = getattr(module, "mem_get_info", None)
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
    except ImportError as e:
        # Do NOT fake a huge value: the KV-cache budget would then attempt a
        # hundreds-of-GB allocation and die with a confusing OOM.
        raise RuntimeError(
            "psutil is required to size the CPU KV cache "
            "(install with the [dev] extra)"
        ) from e


def memory_stats(device: torch.device | None = None) -> dict[str, Any]:
    """Get memory statistics dict for device."""
    if device is None:
        device = get_current_device()
    if device.type in ACCELERATOR_TYPES:
        module = getattr(torch, device.type)
        fn = getattr(module, "memory_stats", None)
        if fn is not None:
            try:
                return fn(device)
            except RuntimeError:
                pass
    return {}


def supports_cuda_graph() -> bool:
    """Whether current device supports CUDA Graph optimization."""
    dtype = get_device_type()
    return dtype in ("cuda", "npu")


# Alias kept for backward compatibility
supports_device_graph = supports_cuda_graph


def get_device_graph_class() -> type:
    """Return the appropriate device graph class for the current device.

    Returns torch.npu.NPUGraph for NPU, torch.cuda.CUDAGraph for CUDA.

    Raises RuntimeError if device graph is not supported.
    """
    dtype = get_device_type()
    if dtype == "cuda":
        import torch

        return torch.cuda.CUDAGraph
    if dtype == "npu":
        import torch

        return torch.npu.NPUGraph
    raise RuntimeError(f"Device graph not supported for device type: {dtype}")


class DeviceGraphContext:
    """Context manager for device graph capture, works with CUDAGraph and NPUGraph.

    NPU requires a non-default stream for capture. We create a dedicated stream.
    """

    def __init__(self, graph: Any, pool: Any = None) -> None:
        self._graph = graph
        self._pool = pool
        self._stream: Any = None
        self._orig_stream: Any = None

    def __enter__(self) -> None:
        import torch

        dtype = get_device_type()
        if dtype == "npu":
            self._orig_stream = torch.npu.current_stream()
            self._stream = torch.npu.Stream()
            torch.npu.set_stream(self._stream)
        self._graph.capture_begin(self._pool)

    def __exit__(self, *args: Any) -> None:
        import torch

        self._graph.capture_end()
        if self._stream is not None:
            torch.npu.set_stream(self._orig_stream)
            self._stream.synchronize()


def get_device_capabilities(device: torch.device | None = None) -> dict[str, Any]:
    """Get capability dict for device."""
    if device is None:
        device = get_current_device()
    dt = device.type
    is_accel = dt in ACCELERATOR_TYPES
    return {
        "device_type": dt,
        "supports_graph": supports_cuda_graph(),
        "supports_empty_cache": is_accel,
        "supports_synchronize": is_accel,
        "supports_memory_stats": is_accel,
    }


def should_use_pin_memory(device: torch.device | None = None) -> bool:
    """Whether pin_memory is beneficial for device."""
    if device is None:
        device = get_current_device()
    return device.type in ("cuda", "npu")


def move_tensor_to_device(
    tensor: torch.Tensor, device: torch.device, non_blocking: bool = False
) -> torch.Tensor:
    """Move tensor to device with consistent error handling."""
    try:
        return tensor.to(device, non_blocking=non_blocking)
    except RuntimeError as e:
        raise RuntimeError(
            f"Cannot move tensor to device {device}. Ensure the device is available."
        ) from e


def is_device_available(device: torch.device) -> bool:
    """Check if a device is available and accessible."""
    dt = device.type
    if dt == "cpu":
        return True
    module = getattr(torch, dt, None)
    if module is None:
        return False
    count_fn = getattr(module, "device_count", None)
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
        raise RuntimeError(
            f"Device {device} is not available. "
            f"Check that the device is properly installed."
        )
