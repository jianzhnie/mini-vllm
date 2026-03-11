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


def create_model(hf_config: Any) -> Any:
    architectures = getattr(hf_config, 'architectures', [])
    model_type = getattr(hf_config, 'model_type', '')

    is_qwen2 = False
    is_opt = False
    if architectures:
        if 'Qwen2ForCausalLM' in architectures:
            is_qwen2 = True
        elif 'OPTForCausalLM' in architectures:
            is_opt = True

    if not is_qwen2 and not is_opt:
        if str(model_type).lower() == 'qwen2':
            is_qwen2 = True
        elif str(model_type).lower() == 'opt':
            is_opt = True

    if is_qwen2:
        return Qwen2ForCausalLM(hf_config)
    if is_opt:
        return OPTForCausalLM(hf_config)
    return Qwen3ForCausalLM(hf_config)
