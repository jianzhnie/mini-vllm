import pytest
import torch
from transformers.utils import (
    is_torch_cuda_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)

from minivllm.utils.device import (
    get_current_device,
    get_default_device_name,
    get_device_capabilities,
    get_device_count,
    get_distributed_backend,
    move_tensor_to_device,
    supports_device_graph,
)


def test_device_detection():
    """Test that devices are detected correctly."""
    # Test device count detection
    device_count = get_device_count()
    assert isinstance(device_count, int)
    assert device_count >= 0

    # Test default device name
    default_device_name = get_default_device_name()
    assert isinstance(default_device_name, str)
    assert default_device_name in [
        'cuda', 'npu', 'xpu', 'mlu', 'musa', 'mps', 'cpu'
    ]

    # Test current device
    current_device = get_current_device()
    assert isinstance(current_device, torch.device)

    # Test device capabilities
    capabilities = get_device_capabilities(current_device)
    assert isinstance(capabilities, dict)
    assert 'device_type' in capabilities
    assert 'supports_graph' in capabilities
    assert 'supports_empty_cache' in capabilities
    assert 'supports_synchronize' in capabilities
    assert 'supports_memory_stats' in capabilities


def test_distributed_backend():
    """Test that the correct distributed backend is returned for each device type."""
    # Test that we get a valid backend string
    backend = get_distributed_backend()
    assert isinstance(backend, str)
    assert backend in ['nccl', 'hccl', 'ccl', 'cncl', 'musa', 'gloo']


def test_tensor_movement():
    """Test that tensors can be moved to different devices correctly."""
    # Create a test tensor
    tensor = torch.randn(10, 10)

    # Test moving to current device
    current_device = get_current_device()
    tensor_on_device = move_tensor_to_device(tensor, current_device)
    assert tensor_on_device.device == current_device

    # Test moving back to CPU
    tensor_on_cpu = move_tensor_to_device(tensor_on_device,
                                          torch.device('cpu'))
    assert tensor_on_cpu.device == torch.device('cpu')


def test_device_specific_features():
    """Test device-specific features are detected correctly."""
    current_device = get_current_device()

    # Test graph support detection
    graph_support = supports_device_graph()
    assert isinstance(graph_support, bool)

    # Test that only CUDA devices support graphs currently
    if current_device.type == 'cuda':
        assert graph_support is True
    else:
        assert graph_support is False

    # Test device capabilities match expected values
    capabilities = get_device_capabilities(current_device)
    if current_device.type in ['cuda', 'npu', 'xpu']:
        assert capabilities['supports_empty_cache'] is True
        assert capabilities['supports_synchronize'] is True
        assert capabilities['supports_memory_stats'] is True
    else:
        assert capabilities['supports_empty_cache'] is False
        assert capabilities['supports_synchronize'] is False
        assert capabilities['supports_memory_stats'] is False


@pytest.mark.skipif(not is_torch_cuda_available(), reason='CUDA not available')
def test_cuda_specific_features():
    """Test CUDA-specific features."""
    if is_torch_cuda_available():
        assert get_device_count() > 0
        assert get_default_device_name() == 'cuda'
        assert get_distributed_backend() == 'nccl'


@pytest.mark.skipif(not is_torch_npu_available(), reason='NPU not available')
def test_npu_specific_features():
    """Test NPU-specific features."""
    if is_torch_npu_available():
        assert get_device_count() > 0
        assert get_default_device_name() == 'npu'
        assert get_distributed_backend() == 'hccl'


@pytest.mark.skipif(not is_torch_xpu_available(), reason='XPU not available')
def test_xpu_specific_features():
    """Test XPU-specific features."""
    if is_torch_xpu_available():
        assert get_device_count() > 0
        assert get_default_device_name() == 'xpu'
        assert get_distributed_backend() == 'ccl'


@pytest.mark.skipif(not is_torch_mlu_available(), reason='MLU not available')
def test_mlu_specific_features():
    """Test MLU-specific features."""
    if is_torch_mlu_available():
        assert get_device_count() > 0
        assert get_default_device_name() == 'mlu'
        assert get_distributed_backend() == 'cncl'


@pytest.mark.skipif(not is_torch_musa_available(), reason='MUSA not available')
def test_musa_specific_features():
    """Test MUSA-specific features."""
    if is_torch_musa_available():
        assert get_device_count() > 0
        assert get_default_device_name() == 'musa'
        assert get_distributed_backend() == 'musa'


@pytest.mark.skipif(not is_torch_mps_available(), reason='MPS not available')
def test_mps_specific_features():
    """Test MPS-specific features."""
    if is_torch_mps_available():
        assert get_device_count() > 0
        assert get_default_device_name() == 'mps'
        assert get_distributed_backend() == 'gloo'
