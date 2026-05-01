from typing import Any

from minivllm.models.manager import ModelManager
from minivllm.models.opt import OPTForCausalLM
from minivllm.models.qwen2 import Qwen2ForCausalLM
from minivllm.models.qwen3 import Qwen3ForCausalLM
from minivllm.utils.logger_utils import get_logger

__all__ = [
    'Qwen2ForCausalLM',
    'Qwen3ForCausalLM',
    'OPTForCausalLM',
    'ModelManager',
    'create_model',
]

logger = get_logger(__name__)

_SUPPORTED_MODELS = {
    'Qwen2ForCausalLM': Qwen2ForCausalLM,
    'Qwen3ForCausalLM': Qwen3ForCausalLM,
    'OPTForCausalLM': OPTForCausalLM,
}

_TYPE_TO_ARCH = {
    'qwen2': 'Qwen2ForCausalLM',
    'qwen3': 'Qwen3ForCausalLM',
    'opt': 'OPTForCausalLM',
}


def create_model(hf_config: Any) -> Any:
    architectures = getattr(hf_config, 'architectures', [])
    model_type = getattr(hf_config, 'model_type', '').lower()

    arch_name = None

    if architectures:
        for arch in architectures:
            if arch in _SUPPORTED_MODELS:
                arch_name = arch
                break

    if arch_name is None and model_type in _TYPE_TO_ARCH:
        arch_name = _TYPE_TO_ARCH[model_type]

    if arch_name is None:
        raise ValueError(
            f'Unsupported model: model_type={model_type!r}, '
            f'architectures={architectures}. '
            f'Supported architectures: {list(_SUPPORTED_MODELS.keys())}')

    logger.info(f'Creating model: {arch_name}')
    return _SUPPORTED_MODELS[arch_name](hf_config)
