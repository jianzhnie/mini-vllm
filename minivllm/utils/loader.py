"""Model weight loading utilities for mini-vllm.

This module provides robust functionality for loading model weights from safetensors files,
supporting both regular parameters and packed modules with custom weight loaders.

Key Features:
    - Loads weights from safetensors files with proper error handling
    - Supports packed modules through optional `packed_modules_mapping` attribute
    - Handles custom weight loaders with flexible argument signatures
    - Provides comprehensive logging and validation
    - Defensive programming with fallback mechanisms

Example:
    >>> from minivllm.utils.loader import load_model
    >>> model = MyModel()
    >>> load_model(model, "./model_weights")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import torch
from safetensors import safe_open
from torch import nn

from minivllm.utils.logger_utils import get_logger

# Configure module logger
logger = get_logger(__name__)

# Type aliases
PackedModulesMapping = dict[str, Tuple[str, Any]]
WeightLoader = Callable[..., None]


@dataclass
class LoadingStats:
    """Statistics for weight loading operation."""
    total_weights: int = 0
    loaded_weights: int = 0
    missing_params: int = 0
    num_files: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate of weight loading."""
        return self.loaded_weights / max(self.total_weights, 1)


def default_weight_loader(param: nn.Parameter,
                          loaded_weight: torch.Tensor) -> None:
    """Load weights by copying tensor data directly to a model parameter.

    This is the fallback loader when no custom weight_loader is defined on a parameter.
    Performs basic validation and copies tensor data while preserving PyTorch properties.

    Args:
        param: The model parameter to load weights into.
        loaded_weight: The tensor containing the weights to load.

    Raises:
        TypeError: If loaded_weight is not a torch.Tensor.
        RuntimeError: If tensor shapes are incompatible.
    """
    if not isinstance(loaded_weight, torch.Tensor):
        raise TypeError(f'Expected torch.Tensor, got {type(loaded_weight)}')

    if param.data.shape != loaded_weight.shape:
        raise RuntimeError(
            f'Shape mismatch: expected {param.data.shape}, got {loaded_weight.shape}'
        )

    param.data.copy_(loaded_weight)


def _find_parameter(model: nn.Module, name: str) -> Optional[nn.Parameter]:
    """Retrieve a parameter by name from the model with fallback mechanisms.

    Supports both custom get_parameter() method and standard named_parameters().

    Args:
        model: The PyTorch module to search.
        name: The parameter name to find.

    Returns:
        The nn.Parameter if found, None otherwise.
    """
    # Try custom get_parameter method first
    if hasattr(model, 'get_parameter') and callable(model.get_parameter):
        try:
            return model.get_parameter(name)
        except Exception as e:
            logger.debug(f"get_parameter failed for '{name}': {e}")

    # Fallback to named_parameters
    return dict(model.named_parameters()).get(name)


def _call_weight_loader(
    weight_loader: WeightLoader,
    param: nn.Parameter,
    tensor: torch.Tensor,
    shard_id: Optional[Any] = None,
) -> None:
    """Call weight loader with flexible argument signatures.

    Args:
        weight_loader: The weight loading function to invoke.
        param: The model parameter.
        tensor: The tensor containing weights.
        shard_id: Optional shard identifier for packed modules.
    """
    try:
        # Try 3-argument form if shard_id provided
        if shard_id is not None:
            weight_loader(param, tensor, shard_id)
            return
    except TypeError:
        pass

    # Fall back to 2-argument form
    weight_loader(param, tensor)


def _validate_path(path: Union[str, Path]) -> Path:
    """Validate and prepare input path for weight loading.

    Args:
        path: Directory path containing safetensors files.

    Returns:
        Validated Path object.

    Raises:
        FileNotFoundError: If path doesn't exist.
        ValueError: If path is not a directory.
    """
    base_path = Path(path)

    if not base_path.exists():
        raise FileNotFoundError(
            f'Model weight directory not found: {base_path}')

    if not base_path.is_dir():
        raise ValueError(f'Path must be a directory: {base_path}')

    return base_path


def _find_safetensor_files(path: Path) -> list[Path]:
    """Find all safetensors files in directory.

    Args:
        path: Directory to search.

    Returns:
        List of safetensors file paths.

    Raises:
        ValueError: If no safetensors files found.
    """
    files = sorted(path.glob('*.safetensors'))
    if not files:
        raise ValueError(f'No .safetensors files found in {path}')
    return files


def _load_single_weight(
    model: nn.Module,
    weight_name: str,
    tensor: torch.Tensor,
    packed_modules_mapping: PackedModulesMapping,
) -> bool:
    """Load a single weight, trying packed mapping first then direct loading.

    Args:
        model: The PyTorch module.
        weight_name: Name of the weight to load.
        tensor: Tensor containing the weight data.
        packed_modules_mapping: Packed modules mapping.

    Returns:
        True if weight was loaded successfully.
    """
    # Try packed modules mapping first
    if _load_with_packed_mapping(model, weight_name, tensor,
                                 packed_modules_mapping):
        return True

    # Fall back to direct loading
    return _load_weight_direct(model, weight_name, tensor)


def _process_safetensors_file(
    model: nn.Module,
    file_path: Path,
    packed_modules_mapping: PackedModulesMapping,
    stats: LoadingStats,
) -> None:
    """Process a single safetensors file.

    Args:
        model: The PyTorch module.
        file_path: Path to safetensors file.
        packed_modules_mapping: Packed modules mapping.
        stats: Loading statistics to update.
    """
    logger.info(f'Loading weights from "{file_path.name}"')

    try:
        with safe_open(str(file_path), framework='pt', device='cpu') as f:
            for weight_name in f.keys():
                stats.total_weights += 1
                tensor = f.get_tensor(weight_name)

                if _load_single_weight(model, weight_name, tensor,
                                       packed_modules_mapping):
                    stats.loaded_weights += 1
                else:
                    stats.missing_params += 1

    except Exception as exc:
        logger.exception(f'Failed to load weights from "{file_path}"')
        raise RuntimeError(
            f'Failed to load weights from "{file_path}": {exc}') from exc


def load_model(model: nn.Module, path: Union[str, Path]) -> None:
    """Load model weights from safetensors files in the specified directory.

    Args:
        model: The PyTorch module to load weights into.
        path: Directory path containing .safetensors files.

    Raises:
        FileNotFoundError: If the specified path doesn't exist.
        ValueError: If path is not a directory or no safetensors files found.
        RuntimeError: If weight loading fails for any safetensors file.
    """
    # Validate path and find files
    base_path = _validate_path(path)
    safetensor_files = _find_safetensor_files(base_path)

    logger.info(
        f'Found {len(safetensor_files)} safetensors file(s) in {base_path}')

    # Get packed modules mapping
    packed_modules_mapping = getattr(model, 'packed_modules_mapping', {}) or {}
    if packed_modules_mapping:
        logger.info(
            f'Using packed modules mapping with {len(packed_modules_mapping)} pattern(s)'
        )

    # Process all files
    stats = LoadingStats(num_files=len(safetensor_files))
    for file_path in safetensor_files:
        _process_safetensors_file(model, file_path, packed_modules_mapping,
                                  stats)

    # Log summary
    _log_loading_summary(stats)


def _load_with_packed_mapping(
    model: nn.Module,
    weight_name: str,
    tensor: torch.Tensor,
    packed_modules_mapping: PackedModulesMapping,
) -> bool:
    """Load weight using packed modules mapping patterns.

    Args:
        model: The PyTorch module.
        weight_name: Name of the weight to load.
        tensor: Tensor containing weight data.
        packed_modules_mapping: Pattern to (replacement_prefix, shard_id) mapping.

    Returns:
        True if pattern matched and weight loaded, False otherwise.
    """
    for pattern, (replacement_prefix,
                  shard_id) in packed_modules_mapping.items():
        if pattern not in weight_name:
            continue

        param_name = weight_name.replace(pattern, replacement_prefix)
        param = _find_parameter(model, param_name)

        if param is None:
            logger.warning(
                f"Parameter '{param_name}' not found for packed weight '{weight_name}' (pattern: '{pattern}')"
            )
            return True

        weight_loader = getattr(param, 'weight_loader', default_weight_loader)
        logger.debug(
            f"Loading packed parameter '{param_name}' (shard_id: {shard_id})")
        _call_weight_loader(weight_loader, param, tensor, shard_id)

        return True

    return False


def _load_weight_direct(
    model: nn.Module,
    weight_name: str,
    tensor: torch.Tensor,
) -> bool:
    """Load weight directly by matching parameter name.

    Args:
        model: The PyTorch module.
        weight_name: Name of the weight/parameter to load.
        tensor: Tensor containing weight data.

    Returns:
        True if weight was loaded successfully, False if parameter not found.
    """
    param = _find_parameter(model, weight_name)
    if param is None:
        logger.warning(
            f"Parameter '{weight_name}' not found in model; skipping weight")
        return False

    weight_loader = getattr(param, 'weight_loader', default_weight_loader)
    logger.debug(f"Loading weight '{weight_name}'")
    _call_weight_loader(weight_loader, param, tensor)
    return True


def _log_loading_summary(stats: LoadingStats) -> None:
    """Log a summary of the weight loading operation.

    Args:
        stats: Loading statistics.
    """
    if stats.missing_params > 0:
        logger.warning(
            f'Weight loading completed with {stats.missing_params} missing parameter(s) out of {stats.total_weights} total weights'
        )
    else:
        logger.info('All weights loaded successfully')

    logger.info(
        f'Loaded {stats.loaded_weights}/{stats.total_weights} weights from {stats.num_files} file(s) (success rate: {stats.success_rate:.1%})'
    )
