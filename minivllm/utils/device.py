import os
from typing import Tuple

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
