import logging
from typing import Any

from minivllm.models.manager import ModelManager
from minivllm.models.opt import OPTForCausalLM
from minivllm.models.qwen2 import Qwen2ForCausalLM
from minivllm.models.qwen3 import Qwen3ForCausalLM

__all__ = [
    'Qwen2ForCausalLM',
    'Qwen3ForCausalLM',
    'OPTForCausalLM',
    'ModelManager',
    'create_model',
]

logger = logging.getLogger(__name__)


def create_model(hf_config: Any) -> Any:
    architectures = getattr(hf_config, 'architectures', [])
    model_type = getattr(hf_config, 'model_type', '').lower()

    is_qwen2 = False
    is_qwen3 = False
    is_opt = False

    if architectures:
        if 'Qwen2ForCausalLM' in architectures:
            is_qwen2 = True
        elif 'Qwen3ForCausalLM' in architectures:
            is_qwen3 = True
        elif 'OPTForCausalLM' in architectures:
            is_opt = True

    if not is_qwen2 and not is_qwen3 and not is_opt:
        if model_type == 'qwen2':
            is_qwen2 = True
        elif model_type == 'qwen3':
            is_qwen3 = True
        elif model_type == 'opt':
            is_opt = True

    if is_qwen2:
        return Qwen2ForCausalLM(hf_config)
    if is_qwen3:
        return Qwen3ForCausalLM(hf_config)
    if is_opt:
        return OPTForCausalLM(hf_config)

    # Default fallback
    logger.warning(
        f'Unknown model type/architecture, defaulting to Qwen3. '
        f'model_type={model_type}, architectures={architectures}')
    return Qwen3ForCausalLM(hf_config)
