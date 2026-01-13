"""Model weight loading utilities for mini-vllm.

Provides functionality for loading model weights from safetensors files with support
for packed modules and custom weight loaders.

Example:
    >>> from minivllm.utils.loader import load_model
    >>> model = MyModel()
    >>> load_model(model, "./model_weights")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import torch
from safetensors import safe_open
from torch import nn

from minivllm.utils.logger_utils import get_logger

logger = get_logger(__name__)

# Type aliases
PackedModulesMapping = dict[str, Tuple[str, Any]]
WeightLoader = Callable[..., None]


def get_default_weight_loader() -> WeightLoader:
    """Get the default weight loader function for copying tensor data."""

    def default_loader(param: nn.Parameter,
                       tensor: torch.Tensor,
                       shard_id: Any = None) -> None:
        if param.data.shape == tensor.shape:
            param.data.copy_(tensor)

    return default_loader


def find_parameter(model: nn.Module, name: str) -> Optional[nn.Parameter]:
    """Find parameter by name with fallback mechanisms."""
    # Try custom get_parameter method first
    if hasattr(model, 'get_parameter') and callable(model.get_parameter):
        try:
            return model.get_parameter(name)
        except Exception as e:
            logger.debug(f"get_parameter failed for '{name}': {e}")

    # Fallback to named_parameters
    return dict(model.named_parameters()).get(name)


def apply_weight_loader(
    param: nn.Parameter,
    tensor: torch.Tensor,
    weight_name: str,
    shard_id: Any = None,
    is_packed: bool = False,
) -> bool:
    """Apply weight loader to a parameter, handling both packed and direct loading.

    Args:
        param: The parameter to load weights into.
        tensor: The tensor containing weight data.
        weight_name: Name of the weight for logging.
        shard_id: Shard ID for packed module loading (if applicable).
        is_packed: Whether this is a packed module load.

    Returns:
        True if loading succeeded, False otherwise.
    """
    weight_loader = getattr(param, 'weight_loader',
                            get_default_weight_loader())

    try:
        if shard_id is not None:
            weight_loader(param, tensor, shard_id)
        else:
            weight_loader(param, tensor)

        log_msg = f"Loaded {'packed ' if is_packed else ''}weight '{weight_name}'"
        if is_packed and shard_id is not None:
            log_msg += f' (shard_id: {shard_id})'
        logger.debug(log_msg)
        return True

    except Exception as e:
        logger.warning(f"Failed to load weight '{weight_name}': {e}")
        return False


def load_weight(
    model: nn.Module,
    weight_name: str,
    tensor: torch.Tensor,
    packed_modules_mapping: PackedModulesMapping,
) -> bool:
    """Load a single weight, trying packed mapping first then direct loading."""
    # Try packed modules mapping first
    for pattern, (replacement_prefix,
                  shard_id) in packed_modules_mapping.items():
        if pattern not in weight_name:
            continue

        param_name = weight_name.replace(pattern, replacement_prefix)
        param = find_parameter(model, param_name)

        if param is None:
            logger.warning(
                f"Parameter '{param_name}' not found for packed weight '{weight_name}'"
            )
            return False

        return apply_weight_loader(param,
                                   tensor,
                                   weight_name,
                                   shard_id,
                                   is_packed=True)

    # Direct loading
    param = find_parameter(model, weight_name)
    if param is None:
        logger.warning(f"Parameter '{weight_name}' not found; skipping weight")
        return False

    return apply_weight_loader(param, tensor, weight_name)


def load_model(model: nn.Module, model_path: Union[str, Path]) -> None:
    """Load model weights from safetensors files in the specified directory.

    Args:
        model: The PyTorch module to load weights into.
        model_path: Directory path containing .safetensors files.

    Raises:
        FileNotFoundError: If the specified path doesn't exist.
        ValueError: If path is not a directory or no safetensors files found.
        RuntimeError: If weight loading fails for any safetensors file.
    """
    # Validate path
    base_path = Path(model_path)
    if not base_path.exists():
        raise FileNotFoundError(
            f'Model weight directory not found: {base_path}')
    if not base_path.is_dir():
        raise ValueError(f'Path must be a directory: {base_path}')

    # Find safetensors files
    safetensor_files = sorted(base_path.glob('*.safetensors'))
    if not safetensor_files:
        raise ValueError(f'No .safetensors files found in {base_path}')

    logger.info(
        f'Found {len(safetensor_files)} safetensors file(s) in {base_path}')

    # Get packed modules mapping
    packed_modules_mapping = getattr(model, 'packed_modules_mapping', {})
    if packed_modules_mapping:
        logger.info(
            f'Using packed modules mapping with {len(packed_modules_mapping)} pattern(s)'
        )

    # Load weights and track statistics
    total_weights = 0
    loaded_weights = 0
    missing_params = 0

    for file_path in safetensor_files:
        logger.info(f'Loading weights from "{file_path.name}"')

        try:
            with safe_open(str(file_path), framework='pt', device='cpu') as f:
                for weight_name in f.keys():
                    total_weights += 1
                    tensor = f.get_tensor(weight_name)

                    if load_weight(model, weight_name, tensor,
                                   packed_modules_mapping):
                        loaded_weights += 1
                    else:
                        missing_params += 1

        except Exception as exc:
            logger.exception(f'Failed to load weights from "{file_path}"')
            raise RuntimeError(
                f'Failed to load weights from "{file_path}": {exc}') from exc

    # Log summary
    success_rate = loaded_weights / max(total_weights, 1)
    if missing_params > 0:
        logger.warning(
            f'Weight loading completed with {missing_params} missing parameter(s) out of {total_weights} total weights'
        )
    else:
        logger.info('All weights loaded successfully')

    logger.info(
        f'Loaded {loaded_weights}/{total_weights} weights from {len(safetensor_files)} file(s) (success rate: {success_rate:.1%})'
    )
